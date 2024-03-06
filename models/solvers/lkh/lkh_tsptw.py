import numpy as np
import scipy
from models.solvers.lkh.lkh_base import LKHBase

class LKHTSPTW(LKHBase):
    def __init__(self, large_value=1e+6, scaling=False, max_trials=1000, seed=1234, lkh_dir="models/solvers/lkh/", io_dir="lkh_io_files"):
        problem = "tsptw"
        super().__init__(problem, large_value, scaling, max_trials, seed, lkh_dir, io_dir)

    def write_data(self, node_feats, f):
        coord_dim = 2
        coords = node_feats["coords"]
        if self.scaling:
            coords = coords * self.large_value
        time_window = node_feats["time_window"].astype(np.int64)
        dist = scipy.spatial.distance.cdist(coords, coords).round().astype(np.int64)
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        f.write("\n".join([
            " ".join(map(str, row))
            for row in dist
        ]))
        f.write("\n")
        f.write("TIME_WINDOW_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, l, u)
            for i, (l, u) in enumerate(time_window)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")