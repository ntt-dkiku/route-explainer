from models.classifiers.ground_truth.ground_truth_base import GroundTruthBase
from models.classifiers.ground_truth.ground_truth_base import get_cap_mask, get_visited_mask, get_tw_mask

class GroundTruthCVRPTW(GroundTruthBase):
    def __init__(self, solver_type):
        problem = "cvrptw"
        compared_problems = ["tsp", "cvrp"]
        super().__init__(problem, compared_problems, solver_type)
    
    # @override
    def get_mask(self, tour, step, node_feats):
        visited = get_visited_mask(tour, step, node_feats)
        less_than_cap = get_cap_mask(tour, step, node_feats)
        not_exceed_tw = get_tw_mask(tour, step, node_feats)
        return visited | (less_than_cap & not_exceed_tw)