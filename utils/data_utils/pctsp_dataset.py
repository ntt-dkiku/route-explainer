import os
import random
import numpy as np
import torch
from utils.utils import load_dataset, save_dataset
from utils.data_utils.dataset_base import DatasetBase, DataLoaderBase
from models.solvers.general_solver import GeneralSolver
from models.classifiers.ground_truth.ground_truth import GroundTruth
from models.classifiers.ground_truth.ground_truth_base import get_visited_mask

class PCTSPDataset(DatasetBase):
    def __init__(self, coord_dim, num_samples, num_nodes, solver="ortools", classifier="ortools", annotation=True, parallel=True, random_seed=1234, num_cpus=os.cpu_count(),
                 penalty_factor=3.):
        """
        Parameters
        ----------
        num_samples: int
            number of samples(instances)
        num_nodes: int 
            number of nodes
        grid_size: int or float32 
            x-pos/y-pos of cities will be in the range [0, grid_size]
        max_tw_gap: 
            maximum time windows gap allowed between the cities constituing the feasible tour
        max_tw_size: 
            time windows of cities will be in the range [0, max_tw_size]
        is_integer_instance: bool 
            True if we want the distances and time widows to have integer values
        seed: int
            seed used for generating the instance. -1 means no seed (instance is random)
        """
        super().__init__(coord_dim, num_samples, num_nodes, annotation, parallel, random_seed, num_cpus)
        self.penalty_factor = penalty_factor
        MAX_LENGTHS = {
            20: 2.,
            50: 3.,
            100: 4.
        }
        self.max_length = MAX_LENGTHS[num_nodes]
        problem = "pctsp"
        solver_type = solver
        classifier_type = classifier
        self.pctsp_solver = GeneralSolver(problem=problem, solver_type=solver_type)
        self.classifier   = GroundTruth(problem=problem, solver_type=classifier_type)

    def generate_instance(self, seed):
        """
        Minor change of https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/pctsp/problem_pctsp.py
        """
        if seed is not None:
            np.random.seed(seed)
        
        #-----------------------------
        # generate locations of nodes
        #-----------------------------
        coords = np.random.uniform(size=(self.num_nodes+1, self.coord_dim))
        
        # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
        # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
        # of the nodes by half of the tour length (which is very rough but similar to op)
        # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
        # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
        # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
        # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
        penalty_max = self.max_length * (self.penalty_factor) / float(self.num_nodes)
        penalties = np.random.uniform(size=(self.num_nodes+1, )) * penalty_max

        # Take uniform prizes
        # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
        # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
        # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
        deterministic_prizes = np.random.uniform(size=(self.num_nodes+1, )) * 4 / float(self.num_nodes)
        deterministic_prizes[0] = 0.0 # Prize at the depot is zero

        return {
            "coords": coords,
            "penalties": penalties,
            "prizes": deterministic_prizes,
            "max_length": np.array([self.max_length]),
            "min_prize": np.min([np.sum(deterministic_prizes), 1.0]),
            "grid_size": np.array([1.0])
        }
    
    def annotate(self, instance):
        # solve PCTSP
        node_feats = instance
        pctsp_tour = self.pctsp_solver.solve(node_feats)
        if pctsp_tour is None:
            return
        # annotate each path
        inputs = self.classifier.get_inputs(pctsp_tour, 0, node_feats)
        labels = self.classifier(inputs, annotation=True)
        if labels is None:
            return
        instance.update({"tour": pctsp_tour, "labels": labels})
        return instance


def get_total_prizes(tour, step, node_feats):
    prizes = node_feats["prizes"]
    total_prize = 0.0
    for i in range(1, step):
        curr_id = tour[i]
        total_prize += prizes[curr_id]
    return total_prize

def get_total_penalty(visited_mask, node_feats):
    penalty = node_feats["penalties"]
    total_penalty = np.sum(penalty[~visited_mask])
    return total_penalty

class PCTSPDataloader(DataLoaderBase):
    # @override
    def load_randomly(self, instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        prizes = torch.FloatTensor(instance["prizes"]) # [num_nodes x 1]
        penalties = torch.FloatTensor(instance["penalties"]) # [num_nodes x 1]
        node_feats = torch.cat((coords, prizes[:, None], penalties[:, None]), -1) # [num_nodes x (coord_dim + 2)]
        tours = instance["tour"]
        labels = instance["labels"]
        for vehicle_id in range(len(labels)):
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                curr_prize   = get_total_prizes(tours[vehicle_id], step, instance)
                curr_penalty = get_total_penalty(visited, instance)
                mask = torch.from_numpy((~visited))
                mask[0] = True # depot is always feasible
                data.append({
                    "node_feats": node_feats,
                    "curr_node_id": torch.tensor(tours[vehicle_id][step-1]).to(torch.long),
                    "next_node_id": torch.tensor(tours[vehicle_id][step]).to(torch.long),
                    "mask": mask,
                    "state": torch.FloatTensor([curr_prize, curr_penalty]),
                    "labels": torch.tensor(label).to(torch.long)
                })
        if fname is not None:
            save_dataset(data, fname, display=False)
            return fname
        else:
            return data

    # @override
    def load_sequentially(self, instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        prizes = torch.FloatTensor(instance["prizes"]) # [num_nodes x 1]
        penalties = torch.FloatTensor(instance["penalties"]) # [num_nodes x 1]
        node_feats = torch.cat((coords, prizes[:, None], penalties[:, None]), -1) # [num_nodes x (coord_dim + 2)]
        tours = instance["tour"]
        labels = instance["labels"]
        num_nodes, node_dim = node_feats.size()
        for vehicle_id in range(len(labels)):
            seq_length = len(labels[vehicle_id])
            curr_node_id_list = []; next_node_id_list = []
            mask_list = []; state_list = []; label_list = []
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                curr_prize   = get_total_prizes(tours[vehicle_id], step, instance)
                curr_penalty = get_total_penalty(visited, instance)
                mask = torch.from_numpy((~visited))
                mask[0] = True # depot is always feasible
                # add values to the lists
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_prize, curr_penalty])
                label_list.append(label)
            data.append({
                "node_feats": node_feats.unsqueeze(0).expand(seq_length, num_nodes, node_dim), # [seq_length x num_nodes x node_feats]
                "curr_node_id": torch.LongTensor(curr_node_id_list), # [seq_length]
                "next_node_id": torch.LongTensor(next_node_id_list), # [seq_length]
                "mask": torch.stack(mask_list, 0), # [seq_length x num_nodes]
                "state": torch.FloatTensor(state_list), # [seq_length x state_dim(1)]
                "labels": torch.LongTensor(label_list)  # [seq_length]
            })
        if fname is not None:
            save_dataset(data, fname, display=False)
            return fname
        else:
            return data

def load_pctsp_sequentially(instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        prizes = torch.FloatTensor(instance["prizes"]) # [num_nodes x 1]
        penalties = torch.FloatTensor(instance["penalties"]) # [num_nodes x 1]
        node_feats = torch.cat((coords, prizes[:, None], penalties[:, None]), -1) # [num_nodes x (coord_dim + 2)]
        tours = instance["tour"]
        labels = instance["labels"]
        num_nodes, node_dim = node_feats.size()
        for vehicle_id in range(len(labels)):
            seq_length = len(tours[vehicle_id])
            curr_node_id_list = []; next_node_id_list = []
            mask_list = []; state_list = []
            for step in range(1, len(tours[vehicle_id])):
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                curr_prize   = get_total_prizes(tours[vehicle_id], step, instance)
                curr_penalty = get_total_penalty(visited, instance)
                mask = torch.from_numpy((~visited))
                mask[0] = True # depot is always feasible
                # add values to the lists
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_prize, curr_penalty])
            data.append({
                "node_feats": node_feats.unsqueeze(0).expand(seq_length, num_nodes, node_dim), # [seq_length x num_nodes x node_feats]
                "curr_node_id": torch.LongTensor(curr_node_id_list), # [seq_length]
                "next_node_id": torch.LongTensor(next_node_id_list), # [seq_length]
                "mask": torch.stack(mask_list, 0), # [seq_length x num_nodes]
                "state": torch.FloatTensor(state_list), # [seq_length x state_dim(1)]
            })
        if fname is not None:
            save_dataset(data, fname, display=False)
            return fname
        else:
            return data