import torch
import torch.nn as nn
import numpy as np
import os
import multiprocessing
from models.solvers.general_solver import GeneralSolver
from utils.utils import calc_tour_length

def get_visited_mask(tour, step, node_feats, dist_matrix=None):
    """
    Visited nodes -> feasible, Unvisited nodes -> infeasible.
    When solving a problem with visited_paths fixed, they should be included to the solution.
    Therefore, visited nodes are set to feasible nodes.
    """
    if dist_matrix is not None:
        num_nodes = len(dist_matrix)
    else:
        num_nodes = len(node_feats["coords"])
    visited = np.isin(np.arange(num_nodes), tour[:step])
    return visited

def get_tw_mask(tour, step, node_feats, dist_matrix=None):
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
    time_window = node_feats["time_window"]
    if dist_matrix is not None:
        num_nodes = len(dist_matrix)
        curr_time = 0.0
        not_exceed_tw = np.ones(num_nodes).astype(np.int32)
        for i in range(1, step):
            prev_id = tour[i - 1]
            curr_id = tour[i]
            travel_time = dist_matrix[prev_id, curr_id]
            # assert curr_time + travel_time < time_window[curr_id, 1], f"Invalid tour! arrival_time: {curr_time + travel_time}, time_window: {time_window[curr_id]}"
            if curr_time + travel_time < time_window[curr_id, 0]:
                curr_time = time_window[curr_id, 0].copy()
            else:
                curr_time += travel_time
        curr_time = curr_time + dist_matrix[tour[step-1]] # [num_nodes] TODO: check
    else:
        coords = node_feats["coords"]
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
        curr_time = curr_time + np.linalg.norm(coords[tour[step-1]][None, :] - coords, axis=-1) # [num_nodes] TODO: check
    not_exceed_tw[curr_time > time_window[:, 1]] = 0
    not_exceed_tw = not_exceed_tw > 0
    return not_exceed_tw

def get_cap_mask(tour, step, node_feats):
    num_nodes = len(node_feats["coords"])
    demands = node_feats["demand"]
    remaining_cap = node_feats["capacity"].copy()
    less_than_cap = np.ones(num_nodes).astype(np.int32)
    for i in range(step):
        remaining_cap -= demands[tour[i]]
    less_than_cap[remaining_cap < demands] = 0
    less_than_cap = less_than_cap > 0
    return less_than_cap

def get_pc_mask(tour, step, node_feats):
    """
    Mask for Price collecting problems (e.g., PCTSP, PCTSPTW, PCCVRP, PCCVRPTW, ...)

    Returns
    -------
    not_exceed_max_length
    """
    large_value = 1e+5
    coords = node_feats["coords"]
    max_length = (node_feats["max_length"] * large_value).astype(np.int64)
    tour_length = 0
    for i in range(1, step):
        prev_id = tour[i - 1]
        curr_id = tour[i]
        tour_length += (np.linalg.norm(coords[prev_id] - coords[curr_id]) * large_value).astype(np.int64)
    curr_to_next  = (np.linalg.norm(coords[tour[step-1]][None, :] - coords, axis=-1) * large_value).astype(np.int64) # [num_nodes]
    next_to_depot = (np.linalg.norm(coords[tour[0]][None, :] - coords, axis=-1) * large_value).astype(np.int64) # [num_nodes]
    not_exceed_max_length = (tour_length + curr_to_next + next_to_depot) <= max_length # [num_nodes]
    return not_exceed_max_length

def analyze_tour(tour, node_feats):
    coords = node_feats["coords"]
    time_window = node_feats["time_window"]
    curr_time = 0
    for i in range(1, len(tour)):
        prev_id = tour[i - 1]
        curr_id = tour[i]
        travel_time = np.linalg.norm(coords[prev_id] - coords[curr_id])
        valid = curr_time + travel_time < time_window[curr_id, 1]
        print(f"visit #{i}: {prev_id} -> {curr_id}, travel_time: {travel_time}, arrival_time: {curr_time + travel_time}, time_window: {time_window[curr_id]}, valid: {valid}")
        if curr_time + travel_time < time_window[curr_id, 0]:
            curr_time = time_window[curr_id, 0]
        else:
            curr_time += travel_time

FAIL_FLAG = -1
class GroundTruthBase(nn.Module):
    def __init__(self, problem, compared_problems, solver_type):
        """
        Parameters
        ----------

        """
        super().__init__()
        self.problem = problem
        self.compared_problems = compared_problems
        self.num_compared_problems = len(compared_problems)
        self.solver_type = solver_type
        self.solvers = []
        for i in range(self.num_compared_problems):
            # TODO:
            self.solvers.append(GeneralSolver(self.compared_problems[i], self.solver_type, scaling=False))

    def forward(self, inputs, annotation=False, parallel=True):
        """
        Parameters
        ----------
        inputs: dict
            tour: 2d list [num_vehicles x seq_length]
            first_explained_step: int
            node_feats: dict of np.array
        annotation: bool
            please set it True when annotating data

        Returns
        -------
        labels: 
        probs:  torch.tensor [batch_size (num_vehicles) x max_seq_length x num_classes]
        """
        input_tours = inputs["tour"]
        node_feats = inputs["node_feats"]
        dist_matrix = inputs["dist_matrix"]
        first_explained_step = inputs["first_explained_step"]
        num_vehicles = len(input_tours)
        if annotation:
            labels = [[] for _ in range(num_vehicles)]
            for vehicle_id in range(num_vehicles):
                input_tour = input_tours[vehicle_id]
                # analyze_tour(input_tour, node_feats)
                for step in range(first_explained_step + 1, len(input_tour)):
                    _, __, label = self.label_path(vehicle_id, step, input_tour, node_feats)
                    if label == FAIL_FLAG:
                        return
                    labels[vehicle_id].append((step, label))
            return labels
        else:
            if parallel:
                labels = [[-1] * (len(range(first_explained_step+1, len(input_tours[vehicle_id])))) for vehicle_id in range(num_vehicles)]
                num_cpus = os.cpu_count()
                with multiprocessing.Pool(num_cpus) as pool:
                    for vehicle_id, step, label in pool.starmap(self.label_path, [(vehicle_id, step, input_tours[vehicle_id], node_feats, dist_matrix)
                                                                                  for vehicle_id in range(num_vehicles) 
                                                                                  for step in range(first_explained_step+1, len(input_tours[vehicle_id]))]):
                        labels[vehicle_id][step-(first_explained_step+1)] = label
            else:
                labels = [[-1] * (len(range(first_explained_step+1, len(input_tours[vehicle_id])))) for vehicle_id in range(num_vehicles)]
                for vehicle_id in range(num_vehicles):
                    for step in range(first_explained_step+1, len(input_tours[vehicle_id])):
                        vehicle_id, step, label = self.label_path(vehicle_id, step, input_tours[vehicle_id], node_feats, dist_matrix)
                        labels[vehicle_id][step-(first_explained_step+1)] = label
            # validate labels
            for vehicle_id in range(num_vehicles):
                assert (len(input_tours[vehicle_id]) - 1) == len(labels[vehicle_id]), f"vehicle_id={vehicle_id}, {input_tours}, {labels}"
            return labels
            # labels = [torch.LongTensor(label) for label in labels] # [num_vehicles x seq_length]
            # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True) # [num_vehicles x max_seq_length]
            # probs = torch.zeros((labels.size(0), labels.size(1), self.num_compared_problems+1)) # [num_vehicles x max_seq_length x (num_compared_problems+1)]
            # probs.scatter_(-1, labels.unsqueeze(-1).expand_as(probs), 1.0)
            # return probs
    
    def label_path(self, vehicle_id, step, input_tour, node_feats, dist_matrix=None):
        compared_tour_list = [[] for _ in range(self.num_compared_problems)]
        visited_path = input_tour[:step].copy()
        new_node_id, new_node_feats, new_dist_matrix = self.get_feasible_nodes(input_tour, step, node_feats, dist_matrix)
        new_visited_path = np.array(list(map(lambda x: np.where(new_node_id==x)[0].item(), visited_path)))
        for i in range(self.num_compared_problems):
            # TODO: in CVRPTW / PCCVRPTW, need to modify classification of the first and last paths
            compared_tours = self.solvers[i].solve(new_node_feats, new_visited_path, new_dist_matrix)
            if compared_tours is None:
                return vehicle_id, step, FAIL_FLAG
            compared_tour = None
            for compared_tour_tmp in compared_tours:
                if new_visited_path[-1] in compared_tour_tmp:
                    compared_tour = compared_tour_tmp
                    break
            assert compared_tour is not None, f"Found no appropriate vhiecle. {compared_tours}, {new_visited_path}"
            compared_tour = np.array(list(map(lambda x: new_node_id[x], compared_tour)))
            if (step > 0) and (compared_tour[1] != input_tour[1]):
                compared_tour = np.flipud(compared_tour) # make direction of the cf tour the same as factual one
            compared_tour_list[i] = compared_tour
        # print("fixed_paths  :", visited_path)
        # print("input_tour   :", input_tour)
        # print("compared_tour:", compared_tour)
        # print()
        # annotation
        label = self.get_label(input_tour, compared_tour_list, step)
        return vehicle_id, step, label

    def solve(self, step, input_tour, node_feats, instance_name=None):
        compared_tours = {} 
        visited_path = input_tour[:step].copy()
        new_node_id, new_node_feats = self.get_feasible_nodes(input_tour, step, node_feats)
        new_visited_path = np.array(list(map(lambda x: np.where(new_node_id==x)[0].item(), visited_path)))
        for i, compared_problem in enumerate(self.compared_problems):
            compared_tours[compared_problem] = self.solvers[i].solve(new_node_feats, new_visited_path, instance_name)
            compared_tours[compared_problem] = list(map(lambda compared_tour: list(map(lambda x: new_node_id[x], compared_tour)), compared_tours[compared_problem]))
            compared_tours[compared_problem] = list(map(lambda compared_tour: calc_tour_length(compared_tour, node_feats["coords"]), compared_tours[compared_problem]))
        return compared_tours

    def get_label(self, input_tour, compared_tours, step):
        for i in range(self.num_compared_problems):
            compared_tour = compared_tours[i]
            if input_tour[step] == compared_tour[step]:
                return i
        return self.num_compared_problems

    def get_inputs(self, tour, first_explained_step, node_feats, dist_matrix=None):
        input_features = {
            "tour": tour, 
            "first_explained_step": first_explained_step, 
            "node_feats": node_feats,
            "dist_matrix": dist_matrix
        }
        return input_features

    def get_feasible_nodes(self, tour, step, node_feats, dist_matrix=None):
        """
        Parameters
        ----------
        tour: np.array [seq_length]
        step: int
        node_feats: np.array [num_nodes x node_dim]

        Returns
        -------
        new_node_id: np.array [num_feasible_nodes]
        new_node_feats: dict of np.array [num_feasible_nodes x coord_dim]
        """
        if dist_matrix is not None:
            num_nodes = len(dist_matrix)
        else:
            num_nodes = len(node_feats["coords"])
        mask = self.get_mask(tour, step, node_feats, dist_matrix)
        node_id = np.arange(num_nodes)
        new_node_id = node_id[mask].copy()
        new_node_feats = {
            key: node_feat[mask].copy() 
                 if key in ["coords", "time_window", "demand", "penalties", "prizes"] else
                 node_feat.copy()
            for key, node_feat in node_feats.items()
        }
        if dist_matrix is not None:
            delete_id = node_id[~mask]
            new_dist_matrix = np.delete(np.delete(dist_matrix, delete_id, 0), delete_id, 1)
        else:
            new_dist_matrix = None
        return new_node_id, new_node_feats, new_dist_matrix

    def get_mask(self, tour, step, node_feats, dist_matrix=None):
        raise NotImplementedError
    
    def check_feasibility(self, tour, node_feats):
        raise NotImplementedError