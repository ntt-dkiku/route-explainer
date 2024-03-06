import torch.nn as nn
from models.solvers.ortools.ortools_tsp import ORToolsTSP
from models.solvers.ortools.ortools_tsptw import ORToolsTSPTW
from models.solvers.ortools.ortools_pctsp import ORToolsPCTSP
from models.solvers.ortools.ortools_pctsptw import ORToolsPCTSPTW
from models.solvers.ortools.ortools_cvrp import ORToolsCVRP
from models.solvers.ortools.ortools_cvrptw import ORToolsCVRPTW

class ORTools(nn.Module):
    def __init__(self, problem, large_value=1e+6, scaling=False):
        super().__init__()
        self.coord_dim = 2
        self.problem = problem
        self.large_value = large_value
        self.scaling = scaling
        self.ortools = self.get_ortools(problem)

    def get_ortools(self, problem):
        """
        Parameters
        ----------
        problem: str
            problem type
        
        Returns
        -------
        ortools: ortools for the specified problem
        """
        if problem == "tsp":
            return ORToolsTSP(self.large_value, self.scaling)
        elif problem == "tsptw":
            return ORToolsTSPTW(self.large_value, self.scaling)
        elif problem == "pctsp":
            return ORToolsPCTSP(self.large_value, self.scaling)
        elif problem == "pctsptw":
            return ORToolsPCTSPTW(self.large_value, self.scaling)
        elif problem == "cvrp":
            return ORToolsCVRP(self.large_value, self.scaling)
        elif problem == "cvrptw":
            return ORToolsCVRPTW(self.large_value, self.scaling)
        else:
            raise NotImplementedError

    def solve(self, node_feats, fixed_paths=None, dist_martix=None, instance_name=None):
        """
        Parameters
        ----------
        node_feats: np.array [num_nodes x node_dim]
        fixed_paths: np.array [cf_step]
        scaling: bool
            whether or not coords are muliplied by a large value
            to convert float-coods into int-coords

        Returns
        -------
        tour: np.array [seq_length]
        """
        return self.ortools.solve(node_feats, fixed_paths, dist_martix, instance_name)