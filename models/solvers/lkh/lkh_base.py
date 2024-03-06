import os
import datetime
import torch.nn as nn
import numpy as np
from subprocess import check_call

NODE_ID_OFFSET = 1

class LKHBase(nn.Module):
    def __init__(self, problem, large_value=1e+6, scaling=False, max_trials=1000, seed=1234, lkh_dir="models/solvers/lkh", io_dir="lkh_io_files"):
        super().__init__()
        self.coord_dim = 2
        self.problem = problem
        self.large_value = large_value
        self.scaling = scaling
        self.max_trials = max_trials
        self.seed = seed

        # I/O file settings
        self.lkh_dir = lkh_dir
        self.io_dir = io_dir
        self.instance_path = f"{io_dir}/{self.problem}/instance"
        self.param_path    = f"{io_dir}/{self.problem}/param"
        self.tour_path     = f"{io_dir}/{self.problem}/tour"
        self.log_path      = f"{io_dir}/{self.problem}/log"
        os.makedirs(self.instance_path, exist_ok=True) 
        os.makedirs(self.param_path, exist_ok=True) 
        os.makedirs(self.tour_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    def solve(self, node_feats, fixed_paths=None, instance_name=None):
        instance_fname = self.write_instance(node_feats, fixed_paths, instance_name)
        param_fname, tour_fname, log_fname = self.write_para(instance_fname, instance_name)
        with open(log_fname, "w") as f:
            check_call([f"{self.lkh_dir}/LKH", param_fname], stdout=f) # run LKH
        tours = self.read_tour(node_feats, tour_fname)
        # clean intermidiate files
        try:
            os.remove(instance_fname); os.remove(param_fname); os.remove(tour_fname); os.remove(log_fname)
        except:
            pass
        return tours
    
    def get_instance_name(self):
        now = datetime.datetime.now()
        instance_name = f"{os.getpid()}-{now.strftime('%Y%m%d_%H%M%S%f')}"
        return instance_name

    def write_instance(self, node_feats, fixed_paths=None, instance_name=None):
        if instance_name is None:
            instance_name = self.get_instance_name()
        instance_fname = f"{self.instance_path}/{instance_name}.{self.problem}"
        with open(instance_fname, "w") as f:
            f.write(f"NAME : {instance_name}\n")
            f.write(f"TYPE : {self.problem.upper()}\n")
            f.write(f"DIMENSION : {len(node_feats['coords'])}\n")
            self.write_data(node_feats, f)
            if fixed_paths is not None and len(fixed_paths) > 1:
                fixed_paths = fixed_paths.copy()
                # FIXED_EDGE_SECTION works well in TSP, but it cannot fix edges in TSPTW
                # EDGE_DATA_SECTION can fix edges in both TSP and TSPTW, but the obtained tour is very poor
                f.write("FIXED_EDGES_SECTION\n")
                fixed_paths += NODE_ID_OFFSET # offset node id (node id starts from 1 in TSPLIB)
                for i in range(len(fixed_paths) - 1):
                    f.write(f"{fixed_paths[i]} {fixed_paths[i+1]}\n")
                # f.write("EDGE_DATA_FORMAT : EDGE_LIST\n")
                # f.write("EDGE_DATA_SECTION\n")
                # avail_edges = self.get_avail_edges(node_feats, fixed_paths)
                # avail_edges += 1 # offset node id (node id starts from 1 in TSPLIB)
                # for i in range(len(avail_edges)):
                #     f.write(f"{avail_edges[i][0]} {avail_edges[i][1]}\n")
            f.write("EOF\n")
        return instance_fname

    def write_data(self, node_feats, f):
        raise NotImplementedError

    def get_avail_edges(self, node_feats, fixed_paths):
        num_nodes = len(node_feats["coords"])
        avail_edges = []
        # add fixed edges
        for i in range(len(fixed_paths) - 1):
            avail_edges.append([fixed_paths[i], fixed_paths[i + 1]])

        # add rest avaialbel edges
        visited = np.array([0] * num_nodes)
        for id in fixed_paths:
            visited[id] = 1
        visited[fixed_paths[0]] = 0
        visited[fixed_paths[-1]] = 0
        mask = visited < 1
        node_id = np.arange(num_nodes)
        feasible_node_id = node_id[mask]
        for j in range(len(feasible_node_id) - 1):
            for i in range(j + 1, len(feasible_node_id)):
                avail_edges.append([feasible_node_id[j], feasible_node_id[i]])
        return np.array(avail_edges)

    def write_para(self, instance_fname, instance_name=None):
        if instance_name is None:
            instance_name = self.get_instance_name()
        param_fname = f"{self.param_path}/{instance_name}.param"
        tour_fname  = f"{self.tour_path}/{instance_name}.tour"
        log_fname   = f"{self.log_path}/{instance_name}.log"
        with open(param_fname, "w") as f:
            f.write(f"PROBLEM_FILE = {instance_fname}\n")
            f.write(f"MAX_TRIALS = {self.max_trials}\n")
            f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 1\n")
            f.write(f"SEED = {self.seed}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_fname}\n")
        return param_fname, tour_fname, log_fname

    def read_tour(self, node_feats, tour_fname):
        """
        Parameters
        ----------
        output_filename: str
            path to a file where optimal tour is written
        Returns
        -------
        tour: 2d list [num_vehicles x seq_length]
            a set of node ids indicating visit order
        """
        if not os.path.exists(tour_fname):
            return # found no feasible solution

        with open(tour_fname, "r") as f:
            tour = []
            is_tour_section = False 
            for line in f:
                line = line.strip()
                if line == "TOUR_SECTION":
                    is_tour_section = True
                    continue
                if is_tour_section:
                    if line != "-1":
                        tour.append(int(line) - NODE_ID_OFFSET)
                    else:
                        tour.append(tour[0])
                        break
        # convert 1d -> 2d list
        num_nodes = len(node_feats["coords"])
        tour = np.array(tour)
        # NOTE: node_id >= num_nodes indicates the depot node. 
        # That is because LKH uses dummy nodes of which locations are the same as the depot and demands = -capacity?
        # I'm not sure where the behavior is documented, but the author of NeuroLKH reads output files like that.
        # please refer to https://github.com/liangxinedu/NeuroLKH/blob/main/CVRPTWdata_generate.py#L132
        tour[tour >= num_nodes] = 0
        # remove subsequent zeros
        tour = tour[np.diff(np.concatenate(([1], tour))).nonzero()]
        loc0 = (tour == 0).nonzero()[0]
        num_vehicles = len(loc0) - 1
        tours = []
        for vehicle_id in range(num_vehicles):
            vehicle_tour = tour[loc0[vehicle_id]:loc0[vehicle_id+1]+1].tolist()
            tours.append(vehicle_tour)
        return tours # offset to make the first index 0
