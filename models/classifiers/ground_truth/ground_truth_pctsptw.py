import numpy as np
from models.classifiers.ground_truth.ground_truth_base import GroundTruthBase
from models.classifiers.ground_truth.ground_truth_base import get_visited_mask, get_tw_mask

class GroundTruthPCTSPTW(GroundTruthBase):
    def __init__(self, solver_type):
        problem = "pctsptw"
        compared_problems = ["tsp", "pctsp"]
        super().__init__(problem, compared_problems, solver_type)
    
    # @override
    def get_mask(self, tour, step, node_feats):
        visited = get_visited_mask(tour, step, node_feats)
        not_exceed_tw = get_tw_mask(tour, step, node_feats)
        return visited | not_exceed_tw