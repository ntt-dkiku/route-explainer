import numpy as np
from models.classifiers.ground_truth.ground_truth_base import GroundTruthBase
from models.classifiers.ground_truth.ground_truth_base import get_visited_mask, get_pc_mask

class GroundTruthPCTSP(GroundTruthBase):
    def __init__(self, solver_type):
        problem = "pctsp"
        compared_problems = ["tsp"]
        super().__init__(problem, compared_problems, solver_type)
    
    # @override
    def get_mask(self, tour, step, node_feats):
        # visited = get_visited_mask(tour, step, node_feats)
        # not_exceed_max_length = get_pc_mask(tour, step, node_feats)
        num_nodes = len(node_feats["coords"])
        return np.full(num_nodes, True)