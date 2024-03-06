import torch.nn as nn
from models.solvers.ortools.ortools import ORTools
from models.solvers.lkh.lkh import LKH
from models.solvers.concorde.concorde import ConcordeTSP

class GeneralSolver(nn.Module):
    def __init__(self, problem, solver_type, large_value=1e+6, scaling=True):
        super().__init__()
        self.problem = problem
        self.large_value = large_value
        self.scaling = scaling
        self.solver_type = solver_type
        supported_problem = {
            "ortools": ["tsp", "tsptw", "pctsp", "pctsptw", "cvrp", "cvrptw"],
            "lkh": ["tsp", "tsptw", "cvrp", "cvrptw"],
            "concorde": ["tsp"]
        }
        # validate solver_type & problem
        assert solver_type in supported_problem.keys(), f"Invalid solver type: {solver_type}. Please select from {supported_problem.keys()}"
        assert problem in supported_problem[solver_type], f"{solver_type} does not support {problem}."
        self.solver = self.get_solver(problem, solver_type)

    def change_solver(self, problem, solver_type):
        if self.solver_type != solver_type or self.problem != problem:
            self.problem = problem
            self.solver_type = solver_type
            self.solver = self.get_solver(problem, solver_type)

    def get_solver(self, problem, solver_type):
        if solver_type == "ortools":
            return ORTools(problem, self.large_value, self.scaling)
        elif solver_type == "lkh":
            return LKH(problem, self.large_value, self.scaling)
        elif solver_type == "concorde":
            assert problem == "tsp", "Concorde solver supports only TSP."
            return ConcordeTSP(self.large_value, self.scaling)
        else:
            assert False, f"Invalid solver type: {solver_type}"

    def solve(self, node_feats, fixed_paths=None, dist_matrix=None, instance_name=None):
        if isinstance(self.solver, ORTools):
            return self.solver.solve(node_feats, fixed_paths, dist_matrix, instance_name)
        else:
            return self.solver.solve(node_feats, fixed_paths, instance_name)