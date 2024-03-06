import torch.nn as nn
import scipy
import numpy as np
import os
import datetime
import subprocess
import models.solvers.concorde.concorde_utils as concorde_utils
import glob
import random

class ConcordeTSP(nn.Module):
    def __init__(self, large_value=1e+6, scaling=False, random_seed=1234, solver_dir="models/solvers/concorde/src/TSP", io_dir="concorde_io_files"):
        self.random_seed = random_seed
        self.large_value = large_value
        self.scaling = scaling
        self.solver_dir = solver_dir
        self.io_dir = io_dir
        self.redirector_stdout = concorde_utils.Redirector(fd=concorde_utils.STDOUT)
        self.redirector_stderr = concorde_utils.Redirector(fd=concorde_utils.STDERR)
        os.makedirs(io_dir, exist_ok=True)

    def get_instance_name(self):
        now = datetime.datetime.now()
        random_value = random.random() # for avoiding duplicated file name
        instance_name = f"{os.getpid()}_{random_value}_{now.strftime('%Y%m%d_%H%M%S%f')}"
        return instance_name
    
    def write_instance(self, node_feats, fixed_paths=None, instance_name=None):
        if instance_name is None:
            instance_name = self.get_instance_name()
        instance_fname = f"{self.io_dir}/{instance_name}.tsp"
        tour_fname = f"{self.io_dir}/{instance_name}.sol"
        with open(instance_fname, "w") as f:
            f.write(f"NAME : {instance_name}\n")
            f.write(f"TYPE : TSP\n")
            f.write(f"DIMENSION : {len(node_feats['coords'])}\n")
            self.write_data(f, node_feats, fixed_paths)
            f.write("EOF\n")
        return instance_fname, tour_fname

    def write_data(self, f, node_feats, fixed_paths=None):
        coords = node_feats["coords"]
        if fixed_paths is None:
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(len(coords)):
                f.write(f" {i + 1} {str(coords[i][0])[:10]} {str(coords[i][1])[:10]}\n")
        else:
            f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            dist = scipy.spatial.distance.cdist(coords, coords).round().astype(np.int64)
            for i in range(len(fixed_paths)):
                curr_id = fixed_paths[i]
                if i != 0 and i != len(fixed_paths) - 1:
                    # NOTE: concorde TSP seems to use int32, so 1e+9 occurs overflow.
                    # 1e+8 could also do the same when N (tour length) is large.
                    dist[curr_id, :] = 1e+8; dist[:, curr_id] = 1e+8
                if i != 0:
                    prev_id = fixed_paths[i - 1]
                    dist[prev_id, curr_id] = 0; dist[curr_id, prev_id] = 0
                if i != len(fixed_paths) - 1:
                    next_id = fixed_paths[i + 1]
                    dist[curr_id, next_id] = 0; dist[next_id, curr_id] = 0
            f.write("\n".join([
                " ".join(map(str, row))
                for row in dist
            ]))

    def solve(self, node_feats, fixed_paths=None, instance_name=None):
        if self.scaling:
            node_feats = self.preprocess_data(node_feats)
        self.redirector_stdout.start()
        self.redirector_stderr.start()
        instance_fname, tour_fname = self.write_instance(node_feats, fixed_paths, instance_name)
        subprocess.run(f"{self.solver_dir}/concorde -o {tour_fname} -x {instance_fname}", shell=True) # run Concorde
        self.redirector_stderr.stop()
        self.redirector_stdout.stop()
        tours = self.read_tour(tour_fname)
        # remove dump (?) files
        try:
            os.remove(instance_fname); os.remove(tour_fname)
        except OSError as e:
            pass
        fname_list = glob.glob("*.sol")
        fname_list.extend(glob.glob("*.res"))
        for fname in fname_list:
            try:
                os.remove(fname)
            except OSError as e:
                # do nothing
                pass
        # subprocess.run(f"rm {instance_name}.sol", shell=True)
        return tours

    def read_tour(self, tour_fname):
        """
        Parameters
        ----------
        tour_fname: str
            path to an output tour
        
        Returns
        -------
        tour: 2d list [num_vehicles(1) x seq_length]
        """
        if not os.path.exists(tour_fname): # fails to solve the instance
            return
        
        tour = []
        with open(tour_fname, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                read_tour = line.split()
                tour.extend(read_tour)
        tour.append(tour[0])
        return [list(map(int, tour))]

    def preprocess_data(self, node_feats):
        # convert float to integer approximately
        return {
            key: (node_feat * self.large_value).astype(np.int64) 
                 if key == "coords" else 
                 node_feat
            for key, node_feat in node_feats.items()
        }
