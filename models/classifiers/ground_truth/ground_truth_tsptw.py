from models.classifiers.ground_truth.ground_truth_base import GroundTruthBase
from models.classifiers.ground_truth.ground_truth_base import get_tw_mask, get_visited_mask

class GroundTruthTSPTW(GroundTruthBase):
    def __init__(self, solver_type):
        problem = "tsptw"
        compared_problems = ["tsp"]
        super().__init__(problem, compared_problems, solver_type)
    
    # @override
    def get_mask(self, tour, step, node_feats, dist_matrix=None):
        visited = get_visited_mask(tour, step, node_feats, dist_matrix)
        not_exceed_tw = get_tw_mask(tour, step, node_feats, dist_matrix)
        return visited | not_exceed_tw