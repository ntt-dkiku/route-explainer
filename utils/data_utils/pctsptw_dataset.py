import os
import random
import numpy as np
import torch
from utils.utils import load_dataset, save_dataset
from scipy.spatial.distance import cdist
from utils.data_utils.dataset_base import DatasetBase, DataLoaderBase
from utils.data_utils.pctsp_dataset import get_total_prizes, get_total_penalty
from utils.data_utils.tsptw_dataset import get_tw_mask2
from models.classifiers.ground_truth.ground_truth_base import get_visited_mask
from models.solvers.general_solver import GeneralSolver
from models.classifiers.ground_truth.ground_truth_pctsptw import GroundTruthPCTSPTW

class PCTSPTWDataset(DatasetBase):
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
        solver_type = solver
        classifier_type = classifier
        problem  = "pctsptw"

        distribution="da_silva"
        max_tw_gap=10
        
        MAX_TW_COEFF = {
            20: 1,
            50: 5,
            100: 10
        }
        self.da_silva_style = distribution == "da_silva"
        self.max_tw_size = MAX_TW_COEFF[num_nodes] * 1000 if self.da_silva_style else 100
        self.max_tw_gap = max_tw_gap
        self.pctsptw_solver = GeneralSolver(problem=problem, solver_type=solver_type)
        self.classifier = GroundTruthPCTSPTW(solver_type=classifier_type)

    def generate_instance(self, seed):
        """
        Minor change of https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/pctsp/problem_pctsp.py
        """
        if seed is not None:
            np.random.seed(seed)
            rand = random.Random()
            rand.seed(seed)

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

        #-------------
        # time window
        #-------------
        # dist = np.sqrt(((coords[0:1] - coords) ** 2).sum(-1)) * 100
        # # define sampling horizon
        # a0 = 0; b0 = 1000
        # a_sample = np.floor(dist) + 1
        # b_sample = b0 - a_sample - 10
        # # sample horizon of each node
        # a = np.random.uniform(size=(self.num_nodes+1,))
        # a = (a * (b_sample - a_sample) + a_sample).astype("int")
        # eps = np.maximum(np.abs(np.random.normal(0, 1, (self.num_nodes+1,))), 0.01)
        # b = np.minimum(np.ceil(a + 300 * eps), b_sample)
        # a[0] = a0; b[0] = b0
        # a = a / 100
        # b = b / 100
        # time_window = np.concatenate((a[:, None], b[:, None]), -1)
        self.grid_size = 100
        random_solution = list(range(1, self.num_nodes+1))
        rand.shuffle(random_solution)
        random_solution = [0] + random_solution # add the depot (node_id=0)
        travel_time = cdist(coords, coords) * self.grid_size # [num_nodes x num_nodes]
        time_windows = np.zeros((self.num_nodes+1, 2))
        time_windows[0, :] = [0, 1000 * self.grid_size] # time window for the depot
        total_dist = 0
        for i in range(1, self.num_nodes+1):
            prev_node_id = random_solution[i-1]
            cur_node_id = random_solution[i]

            cur_dist = travel_time[prev_node_id][cur_node_id]

            tw_lb_min = time_windows[prev_node_id, 0] + cur_dist
            total_dist += cur_dist

            if self.da_silva_style:
                # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
                rand_tw_lb = rand.uniform(total_dist - self.max_tw_size / 2, total_dist)
                rand_tw_ub = rand.uniform(total_dist, total_dist + self.max_tw_size / 2)
            else:
                # Cappart et al. style 'propagates' the time windows resulting in little overlap / easier instances
                rand_tw_lb = rand.uniform(tw_lb_min, tw_lb_min + self.max_tw_gap)
                rand_tw_ub = rand.uniform(rand_tw_lb, rand_tw_lb + self.max_tw_size)

            time_windows[cur_node_id, :] = [rand_tw_lb, rand_tw_ub] # [num_nodes x 2(start, end)]

        return {
            "coords": coords,
            "penalties": penalties,
            "prizes": deterministic_prizes,
            "time_window": time_windows / self.grid_size,
            "min_prize": np.min([np.sum(deterministic_prizes), 1.0]),
            "grid_size": np.array([1.0])
        }
    
    def annotate(self, instance):
        # solve PCTSPTW
        node_feats = instance
        pctsptw_tour = self.pctsptw_solver.solve(node_feats)
        # print(pctsptw_tour)
        if pctsptw_tour is None:
            return
        # annotate each path
        inputs = self.classifier.get_inputs(pctsptw_tour, 0, node_feats)
        labels = self.classifier(inputs, annotation=True)
        if labels is None:
            return
        instance.update({"tour": pctsptw_tour, "labels": labels})
        return instance
    

class PCTSPTWDataloader(DataLoaderBase):
    # @override
    def load_randomly(self, instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        prizes = torch.FloatTensor(instance["prizes"]) # [num_nodes x 1]
        penalties = torch.FloatTensor(instance["penalties"]) # [num_nodes x 1]
        raw_time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0)
        time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0) # [num_nodes x 2]
        time_window = (time_window - time_window[1:].min()) / (time_window[1:].max() - time_window[1:].min()) # min-max normalization
        node_feats = torch.cat((coords, prizes[:, None], penalties[:, None], time_window), -1) # [num_nodes x (coord_dim + 4)]
        tours = instance["tour"]
        labels = instance["labels"]
        for vehicle_id in range(len(labels)):
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                curr_prize   = get_total_prizes(tours[vehicle_id], step, instance)
                curr_penalty = get_total_penalty(visited, instance)
                not_exceed_tw, curr_time = get_tw_mask2(tours[vehicle_id], step, instance)
                curr_time = ((curr_time - raw_time_window[1:].min()) / (raw_time_window[1:].max() - raw_time_window[1:].min())).item()
                mask = torch.from_numpy((~visited) & not_exceed_tw)
                mask[0] = True # depot is always feasible
                data.append({
                    "node_feats": node_feats,
                    "curr_node_id": torch.tensor(tours[vehicle_id][step-1]).to(torch.long),
                    "next_node_id": torch.tensor(tours[vehicle_id][step]).to(torch.long),
                    "mask": mask,
                    "state": torch.FloatTensor([curr_prize, curr_penalty, curr_time]),
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
        raw_time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0)
        time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0) # [num_nodes x 2]
        time_window = (time_window - time_window[1:].min()) / (time_window[1:].max() - time_window[1:].min()) # min-max normalization
        node_feats = torch.cat((coords, prizes[:, None], penalties[:, None], time_window), -1) # [num_nodes x (coord_dim + 4)]
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
                not_exceed_tw, curr_time = get_tw_mask2(tours[vehicle_id], step, instance)
                curr_time = ((curr_time - raw_time_window[1:].min()) / (raw_time_window[1:].max() - raw_time_window[1:].min())).item()
                mask = torch.from_numpy((~visited) & not_exceed_tw)
                mask[0] = True # depot is always feasible
                # add values to the lists
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_prize, curr_penalty, curr_time])
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

def load_pctsptw_sequentially(instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        prizes = torch.FloatTensor(instance["prizes"]) # [num_nodes x 1]
        penalties = torch.FloatTensor(instance["penalties"]) # [num_nodes x 1]
        raw_time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0)
        time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0) # [num_nodes x 2]
        time_window = (time_window - time_window[1:].min()) / (time_window[1:].max() - time_window[1:].min()) # min-max normalization
        node_feats = torch.cat((coords, prizes[:, None], penalties[:, None], time_window), -1) # [num_nodes x (coord_dim + 4)]
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
                not_exceed_tw, curr_time = get_tw_mask2(tours[vehicle_id], step, instance)
                curr_time = ((curr_time - raw_time_window[1:].min()) / (raw_time_window[1:].max() - raw_time_window[1:].min())).item()
                mask = torch.from_numpy((~visited) & not_exceed_tw)
                mask[0] = True # depot is always feasible
                # add values to the lists
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_prize, curr_penalty, curr_time])
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