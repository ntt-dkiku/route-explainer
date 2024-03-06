import numpy as np
import scipy
from models.solvers.lkh.lkh_base import LKHBase

class LKHCVRP(LKHBase):
    def __init__(self, large_value=1e+6, scaling=False, max_trials=1000, seed=1234, lkh_dir="models/solvers/lkh/", io_dir="lkh_io_files"):
        problem = "cvrp"
        super().__init__(problem, large_value, scaling, max_trials, seed, lkh_dir, io_dir)
    
    def write_data(self, node_feats, f):
        """
        Paramters
        ---------
        node_feats: dict of np.array 
            coords: np.array [num_nodes x coord_dim]
            demand: np.array [num_nodes x 1]
            capacity: np.array [1]
        """
        coords = node_feats["coords"]
        demand = node_feats["demand"]
        capacity = node_feats["capacity"][0]
        num_nodes = len(coords)
        if self.scaling:
            coords = coords * self.large_value
        # NOTE: In CVRP, LKH can automatically obtain optimal vehicle size.
        # However it cannot in CVRPTW (please check lkh_cvrptw.py).
        # EDGE_WEIGHT_SECTION
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        # CAPACITY
        f.write("CAPACITY : " + str(capacity) + "\n")
        # NODE_COORD_SECTION
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_nodes):
            f.write(f" {i + 1} {str(coords[i][0])[:10]} {str(coords[i][1])[:10]}\n")
        # DEMAND_SECTION
        f.write("DEMAND_SECTION\n")
        for i in range(num_nodes):
            f.write(f" {i + 1} {str(demand[i])}\n")
        # DEPOT SECTION
        f.write("DEPOT_SECTION\n")
        f.write("1\n")