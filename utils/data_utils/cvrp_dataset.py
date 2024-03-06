import os
import numpy as np
import torch
from utils.utils import load_dataset, save_dataset
from utils.data_utils.dataset_base import DatasetBase, DataLoaderBase
from models.solvers.general_solver import GeneralSolver
from models.classifiers.ground_truth.ground_truth_base import get_visited_mask
from models.classifiers.ground_truth.ground_truth_cvrp import GroundTruthCVRP

class CVRPDataset(DatasetBase):
    def __init__(self, coord_dim, num_samples, num_nodes, solver="ortools", classifier="ortools", annotation=True, parallel=True, random_seed=1234, num_cpus=os.cpu_count()):
        super().__init__(coord_dim, num_samples, num_nodes, annotation, parallel, random_seed, num_cpus)
        CAPACITY = {
            10: 20,
            20: 30,
            50: 40,
            100: 50
        }
        self.capacity = CAPACITY[num_nodes]
        problem = "cvrp"
        solver_type = solver
        classifier_solver = classifier
        self.cvrp_solver = GeneralSolver(problem=problem, solver_type=solver_type)
        self.classifier = GroundTruthCVRP(solver_type=classifier_solver)

    def generate_instance(self, seed):
        np.random.seed(seed)
        coords = np.random.uniform(size=(self.num_nodes+1, self.coord_dim))
        demand = np.random.randint(1, 10, size=(self.num_nodes+1, ))
        demand[0] = 0 # set demand of the depot to zero
        return {
            "coords": coords,
            "demand": demand,
            "grid_size": np.array([1.0]),
            "capacity": np.array([self.capacity], dtype=np.int64)
        }

    def annotate(self, instance):
        """
        Paramters
        ---------
        """
        # solve CVRP
        node_feats = instance
        cvrp_tours = self.cvrp_solver.solve(node_feats)
        if cvrp_tours is None:
            return
        inputs = self.classifier.get_inputs(cvrp_tours, 0, node_feats)
        labels = self.classifier(inputs, annotation=True)
        if labels is None:
            return
        instance.update({"tour": cvrp_tours, "labels": labels}) 
        return instance

    def get_feasible_nodes(self):
        pass


def get_cap_mask2(tour, step, node_feats):
    num_nodes = len(node_feats["coords"])
    demands = node_feats["demand"]
    remaining_cap = node_feats["capacity"].copy().item()
    less_than_cap = np.ones(num_nodes).astype(np.int32)
    for i in range(step):
        remaining_cap -= demands[tour[i]]
    less_than_cap[remaining_cap < demands] = 0
    less_than_cap = less_than_cap > 0
    return less_than_cap, (remaining_cap / node_feats["capacity"].item())

class CVRPDataloader(DataLoaderBase):
    # @override
    def load_randomly(self, instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"])  # [num_nodes x coord_dim]
        demands = torch.FloatTensor(instance["demand"] / instance["capacity"]) # [num_nodes x 1]
        node_feats = torch.cat((coords, demands[:, None]), -1)   # [num_nodes x (coord_dim + 1)]
        tours = instance["tour"]
        labels = instance["labels"]
        for vehicle_id in range(len(labels)):
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                not_exceed_cap, curr_cap = get_cap_mask2(tours[vehicle_id], step, instance)
                mask = torch.from_numpy((~visited) & not_exceed_cap)
                mask[0] = True # depot is always feasible
                data.append({
                    "node_feats": node_feats,
                    "curr_node_id": torch.tensor(tours[vehicle_id][step-1]).to(torch.long),
                    "next_node_id": torch.tensor(tours[vehicle_id][step]).to(torch.long),
                    "mask": mask,
                    "state": torch.FloatTensor([curr_cap]),
                    "labels": torch.tensor(label).to(torch.long)
                })
        if fname is not None:
            save_dataset(data, fname, display=False)
            return fname
        else:
            return data

    def load_sequentially(self, instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        demands = torch.FloatTensor(instance["demand"] / instance["capacity"])# [num_nodes x 1]
        node_feats = torch.cat((coords, demands[:, None]), -1)   # [num_nodes x (coord_dim + 1)]
        tours = instance["tour"]
        labels = instance["labels"]
        num_nodes, node_dim = node_feats.size()
        for vehicle_id in range(len(labels)):
            seq_length = len(labels[vehicle_id])
            curr_node_id_list = []; next_node_id_list = []
            mask_list = []; state_list = []; label_list_ = []
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                not_exceed_cap, curr_cap = get_cap_mask2(tours[vehicle_id], step, instance)
                mask = torch.from_numpy((~visited) & not_exceed_cap)
                mask[0] = True # depot is always feasible
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_cap])
                label_list_.append(label)
            data.append({
                "node_feats": node_feats.unsqueeze(0).expand(seq_length, num_nodes, node_dim), # [seq_length x num_nodes x node_feats]
                "curr_node_id": torch.LongTensor(curr_node_id_list), # [seq_length]
                "next_node_id": torch.LongTensor(next_node_id_list), # [seq_length]
                "mask": torch.stack(mask_list, 0), # [seq_length x num_nodes]
                "state": torch.FloatTensor(state_list), # [seq_length x state_dim(1)]
                "labels": torch.LongTensor(label_list_)  # [seq_length]
            })
        if fname is not None:
            save_dataset(data, fname, display=False)
            return fname
        else:
            return data
        
def load_cvrp_sequentially(instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        demands = torch.FloatTensor(instance["demand"] / instance["capacity"])# [num_nodes x 1]
        node_feats = torch.cat((coords, demands[:, None]), -1)   # [num_nodes x (coord_dim + 1)]
        tours = instance["tour"]
        labels = instance["labels"]
        num_nodes, node_dim = node_feats.size()
        for vehicle_id in range(len(labels)):
            seq_length = len(tours[vehicle_id])
            curr_node_id_list = []; next_node_id_list = []
            mask_list = []; state_list = []
            for step in range(1, len(tours[vehicle_id])):
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                not_exceed_cap, curr_cap = get_cap_mask2(tours[vehicle_id], step, instance)
                mask = torch.from_numpy((~visited) & not_exceed_cap)
                mask[0] = True # depot is always feasible
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_cap])
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