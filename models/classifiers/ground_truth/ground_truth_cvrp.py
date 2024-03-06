from models.classifiers.ground_truth.ground_truth_base import GroundTruthBase
from models.classifiers.ground_truth.ground_truth_base import get_cap_mask, get_visited_mask

class GroundTruthCVRP(GroundTruthBase):
    def __init__(self, solver_type):
        problem = "cvrp"
        compared_problems = ["tsp"]
        super().__init__(problem, compared_problems, solver_type)
    
    # @override
    def get_mask(self, tour, step, node_feats):
        visited = get_visited_mask(tour, step, node_feats)
        less_than_cap = get_cap_mask(tour, step, node_feats)
        return visited | less_than_cap