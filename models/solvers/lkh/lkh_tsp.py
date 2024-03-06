from models.solvers.lkh.lkh_base import LKHBase

class LKHTSP(LKHBase):
    def __init__(self, large_value=1e+6, scaling=False, max_trials=1000, seed=1234, lkh_dir="models/solvers/lkh/", io_dir="lkh_io_files"):
        problem = "tsp"
        super().__init__(problem, large_value, scaling, max_trials, seed, lkh_dir, io_dir)

    def write_data(self, node_feats, f):
        coords = node_feats["coords"]
        if self.scaling:
            coords = coords * self.large_value
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(len(coords)):
            f.write(f" {i + 1} {str(coords[i][0])[:10]} {str(coords[i][1])[:10]}\n")