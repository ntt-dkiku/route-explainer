import os
import random
from tqdm import tqdm
import multiprocessing
from utils.utils import save_dataset
import numpy as np
import torch
from scipy.spatial.distance import cdist
from utils.utils import load_dataset
from utils.data_utils.dataset_base import DatasetBase, DataLoaderBase
from models.solvers.general_solver import GeneralSolver
from models.classifiers.ground_truth.ground_truth_base import get_tw_mask, get_visited_mask
from models.classifiers.ground_truth.ground_truth_tsptw import GroundTruthTSPTW

class TSPTWDataset(DatasetBase):
    def __init__(self, coord_dim, num_samples, num_nodes, solver="ortools", classifier="ortools", annotation=True, parallel=True, random_seed=1234, num_cpus=os.cpu_count(),
                 grid_size=100, is_integer_instance=False, distribution="da_silva", max_tw_gap=10):
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
        self.grid_size = grid_size
        self.is_integer_instance = is_integer_instance
        self.da_silva_style = distribution == "da_silva"
        self.max_tw_size = 1000 if self.da_silva_style else 100
        self.max_tw_gap = max_tw_gap
        solver_type = solver
        classifier_type = classifier
        self.tsptw_solver = GeneralSolver(problem="tsptw", solver_type=solver_type)
        self.classifier = GroundTruthTSPTW(solver_type=classifier_type)

    def generate_instance(self, seed):
        """
        Parameters
        ----------
        seed: int
            random seed

        Returns
        --------
        a feasible TSPTW instance randomly generated using the parameters
        -------
        """

        rand = random.Random()
        if seed is not None:
            rand.seed(seed)
            np.random.seed(seed)

        #-----------------------------
        # generate locations of nodes
        #-----------------------------
        coords = np.random.uniform(size=(self.num_nodes, self.coord_dim))

        #-------------------------------------------------------------------------------
        # compute travel time b/w two nodes, which is identical to distance b/w the two
        #-------------------------------------------------------------------------------
        travel_time = cdist(coords, coords) * self.grid_size # [num_nodes x num_nodes]
        if self.is_integer_instance:
            travel_time = travel_time.round().astype(np.int64)

        #------------------------------------------------------------------
        # generate a random tour to guarantee existence of a fieasble tour
        #------------------------------------------------------------------
        random_solution = list(range(1, self.num_nodes))
        rand.shuffle(random_solution)
        random_solution = [0] + random_solution # add the depot (node_id=0)

        #----------------------
        # generate time window
        #----------------------
        time_windows = np.zeros((self.num_nodes, 2))
        time_windows[0, :] = [0, 100 * self.grid_size] # time window for the depot
        total_dist = 0
        for i in range(1, self.num_nodes):
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

            if self.is_integer_instance:
                rand_tw_lb = np.floor(rand_tw_lb)
                rand_tw_ub = np.ceil(rand_tw_ub)

            time_windows[cur_node_id, :] = [rand_tw_lb, rand_tw_ub] # [num_nodes x 2(start, end)]

        if self.is_integer_instance:
            time_windows = time_windows.astype(np.int64)

        # Don't store travel time since it takes up much
        return {
            "coords": coords,
            "time_window": time_windows / self.grid_size,
            "grid_size": np.array([1.0])
        }
    
    def annotate(self, instance):
        """
        Paramters
        ---------
        instance: dict
            coords: np.array [num_nodes x coord_dim]
            time_window: np.array [num_nodes x 2(start, end)]
            grid_size: int or float32
        
        Returns
        -------
        labeled instance: dict
            coords: np.array [num_nodes x coord_dim]
            time_window: np.array [num_nodes x 2(start, end)]
            grid_size: int or float32
            tour: np.array [seq_length]
            labels: 2d list [num_labeled_step x 2(step, label)]
        """
        # solve TSPTW
        num_nodes = len(instance["coords"])
        node_feats = instance
        tsptw_tour = self.tsptw_solver.solve(node_feats)
        if len(tsptw_tour[0]) != num_nodes + 1:
            # print("Could not find a feasible tour! Skip current instance.")
            return
        # annotate each path
        inputs = self.classifier.get_inputs(tsptw_tour, 0, node_feats)
        labels = self.classifier(inputs, annotation=True)
        if labels is None:
            return
        instance.update({"tour": tsptw_tour, "labels": labels})
        return instance

def get_tw_mask2(tour, step, node_feats):
    """
    Nodes whose tw exceeds current_time -> infeasible, otherwise -> feasible.

    Parameters
    ----------
    tour: list [seq_length]
    step: int
    node_feats: dict of np.array

    Returns
    -------
    mask_tw: np.array [num_nodes]
    """
    node_feats = node_feats.copy()
    coords = node_feats["coords"]
    time_window = node_feats["time_window"]
    num_nodes = len(coords)
    curr_time = 0.0
    not_exceed_tw = np.ones(num_nodes).astype(np.int32)
    for i in range(1, step):
        prev_id = tour[i - 1]
        curr_id = tour[i]
        travel_time = np.linalg.norm(coords[prev_id] - coords[curr_id])
        # assert curr_time + travel_time < time_window[curr_id, 1], f"Invalid tour! arrival_time: {curr_time + travel_time}, time_window: {time_window[curr_id]}"
        if curr_time + travel_time < time_window[curr_id, 0]:
            curr_time = time_window[curr_id, 0].copy()
        else:
            curr_time += travel_time
    next_time = curr_time + np.linalg.norm(coords[tour[step-1]][None, :] - coords, axis=-1) # [num_nodes] TODO: check
    not_exceed_tw[next_time > time_window[:, 1]] = 0
    not_exceed_tw = not_exceed_tw > 0
    return not_exceed_tw, curr_time


class TSPTWDataloader(DataLoaderBase):
    # @override
    def load_randomly(self, instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        raw_time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0)
        time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0) # [num_nodes x 2]
        time_window = (time_window - time_window[1:].min()) / (time_window[1:].max() - time_window[1:].min()) # min-max normalization
        node_feats = torch.cat((coords, time_window), -1) # [num_nodes x (coord_dim + 2)]
        tours = instance["tour"]
        labels = instance["labels"]
        for vehicle_id in range(len(labels)):
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                not_exceed_tw, curr_time = get_tw_mask2(tours[vehicle_id], step, instance)
                curr_time = ((curr_time - raw_time_window[1:].min()) / (raw_time_window[1:].max() - raw_time_window[1:].min())).item()
                mask = torch.from_numpy((~visited) & not_exceed_tw)
                mask[0] = True # depot is always feasible
                data.append({
                    "node_feats": node_feats,
                    "curr_node_id": torch.tensor(tours[vehicle_id][step-1]).to(torch.long),
                    "next_node_id": torch.tensor(tours[vehicle_id][step]).to(torch.long),
                    "mask": mask,
                    "state": torch.FloatTensor([curr_time]),
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
        raw_time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0)
        time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0) # [num_nodes x 2]
        time_window = (time_window - time_window[1:].min()) / (time_window[1:].max() - time_window[1:].min()) # min-max normalization
        node_feats = torch.cat((coords, time_window), -1) # [num_nodes x (coord_dim + 2)]
        tours = instance["tour"]
        labels = instance["labels"]
        num_nodes, node_dim = node_feats.size()
        for vehicle_id in range(len(labels)):
            seq_length = len(labels[vehicle_id])
            curr_node_id_list = []; next_node_id_list = []
            mask_list = []; state_list = []; label_list_ = []
            for step, label in labels[vehicle_id]:
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                not_exceed_tw, curr_time = get_tw_mask2(tours[vehicle_id], step, instance)
                curr_time = (curr_time - raw_time_window[1:].min()) / (raw_time_window[1:].max() - raw_time_window[1:].min()).item()
                mask = torch.from_numpy((~visited) & not_exceed_tw)
                mask[0] = True # depot is always feasible
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_time])
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

def load_tsptw_sequentially(instance, fname=None):
        data = []
        coords = torch.FloatTensor(instance["coords"]) # [num_nodes x coord_dim]
        raw_time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0)
        time_window = torch.FloatTensor(instance["time_window"]).clamp(0.0) # [num_nodes x 2]
        time_window = (time_window - time_window[1:].min()) / (time_window[1:].max() - time_window[1:].min()) # min-max normalization
        node_feats = torch.cat((coords, time_window), -1) # [num_nodes x (coord_dim + 2)]
        tours = instance["tour"]
        labels = instance["labels"]
        num_nodes, node_dim = node_feats.size()
        for vehicle_id in range(len(labels)):
            seq_length = len(tours[vehicle_id])
            curr_node_id_list = []; next_node_id_list = []
            mask_list = []; state_list = []
            for step in range(1, len(tours[vehicle_id])):
                visited = get_visited_mask(tours[vehicle_id], step, instance)
                not_exceed_tw, curr_time = get_tw_mask2(tours[vehicle_id], step, instance)
                curr_time = (curr_time - raw_time_window[1:].min()) / (raw_time_window[1:].max() - raw_time_window[1:].min()).item()
                mask = torch.from_numpy((~visited) & not_exceed_tw)
                mask[0] = True # depot is always feasible
                curr_node_id_list.append(tours[vehicle_id][step-1])
                next_node_id_list.append(tours[vehicle_id][step])
                mask_list.append(mask)
                state_list.append([curr_time])
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