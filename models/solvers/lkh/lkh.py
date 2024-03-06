import torch.nn as nn
from models.solvers.lkh.lkh_tsp import LKHTSP
from models.solvers.lkh.lkh_tsptw import LKHTSPTW
from models.solvers.lkh.lkh_cvrp import LKHCVRP
from models.solvers.lkh.lkh_cvrptw import LKHCVRPTW

class LKH(nn.Module):
    def __init__(self, problem, large_value=1e+6, scaling=False, max_trials=10, seed=1234, lkh_dir="models/solvers/lkh/src", io_dir="lkh_io_files"):
        super().__init__()
        self.probelm = problem
        if problem == "tsp":
            self.lkh = LKHTSP(large_value, scaling, max_trials, seed, lkh_dir, io_dir)
        elif problem == "tsptw":
            self.lkh = LKHTSPTW(large_value, scaling, max_trials, seed, lkh_dir, io_dir)
        elif problem == "cvrp":
            self.lkh = LKHCVRP(large_value, scaling, max_trials, seed, lkh_dir, io_dir)
        elif problem == "cvrptw":
            self.lkh = LKHCVRPTW(large_value, scaling, max_trials, seed, lkh_dir, io_dir)
    
    def solve(self, node_feats, fixed_paths=None, instance_name=None):
        return self.lkh.solve(node_feats, fixed_paths, instance_name)
