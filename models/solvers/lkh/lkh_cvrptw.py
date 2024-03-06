import numpy as np
import scipy
from models.solvers.lkh.lkh_base import LKHBase

class LKHCVRPTW(LKHBase):
    def __init__(self, large_value=1e+6, scaling=False, max_trials=1000, seed=1234, lkh_dir="models/solvers/lkh/", io_dir="lkh_io_files"):
        problem = "cvrptw"
        super().__init__(problem, large_value, scaling, max_trials, seed, lkh_dir, io_dir)
    
    def write_data(self, node_feats, f):
        """
        Paramters
        ---------
        node_feats: dict of np.array 
            coords: np.array [num_nodes x coord_dim]
            demand: np.array [num_nodes x 1]
            capacity: np.array [1]
            time_window: np.array [num_nodes x 2(start, end)]
        """
        coords = node_feats["coords"]
        demand = node_feats["demand"]
        capacity = node_feats["capacity"][0]
        time_window = node_feats["time_window"]
        num_nodes = len(coords)
        if self.scaling:
            coords = coords * self.large_value
            time_window = time_window * self.large_value
        # VEHICLES
        # As the number of unused vehicles is also included to penalty in default,
        # we have to modify Penalty_CVRTW.c in LKH SRC directory.
        # Comment out the following part, which corresponds to penaly of unsed vehicles:
        #   42 if (MTSPMinSize >= 1 && Size < MTSPMinSize)
        #   43    P += MTSPMinSize - Size;
        #   44 if (Size > MTSPMaxSize)
        #   45    P += Size - MTSPMaxSize;
        # After the modification, we can automatically obtain optimal vehicle size 
        # by setting large vehicle size (e.g. >20) here
        f.write("VEHICLES : 20\n")
        # CAPACITY
        f.write("CAPACITY : " + str(capacity) + "\n")
        # EDGE_WEIGHT_SECTION
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        # NODE_COORD_SECTION
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_nodes):
            f.write(f" {i + 1} {str(coords[i][0])[:10]} {str(coords[i][1])[:10]}\n")
        # DEMAND_SECTION
        f.write("DEMAND_SECTION\n")
        for i in range(num_nodes):
            f.write(f" {i + 1} {str(demand[i])}\n")
        # TIME_WINDOW_SECTION
        f.write("TIME_WINDOW_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, l, u)
            for i, (l, u) in enumerate(time_window)
        ]))
        # DEPOT SECTION
        f.write("DEPOT_SECTION\n")
        f.write("1\n")