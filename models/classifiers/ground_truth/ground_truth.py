import torch
import torch.nn as nn
import numpy as np
from models.classifiers.ground_truth.ground_truth_tsptw import GroundTruthTSPTW
from models.classifiers.ground_truth.ground_truth_pctsp import GroundTruthPCTSP
from models.classifiers.ground_truth.ground_truth_pctsptw import GroundTruthPCTSPTW
from models.classifiers.ground_truth.ground_truth_cvrp import GroundTruthCVRP
from models.classifiers.ground_truth.ground_truth_cvrptw import GroundTruthCVRPTW

class GroundTruth(nn.Module):
    def __init__(self, problem, solver_type):
        super().__init__()
        self.problem = problem
        self.solver_type = solver_type
        if problem == "tsptw":
            self.ground_truth = GroundTruthTSPTW(solver_type)
        elif problem == "pctsp":
            self.ground_truth = GroundTruthPCTSP(solver_type)
        elif problem == "pctsptw":
            self.ground_truth = GroundTruthPCTSPTW(solver_type)
        elif problem == "cvrp":
            self.ground_truth = GroundTruthCVRP(solver_type)
        elif problem == "cvrptw":
            self.ground_truth = GroundTruthCVRPTW(solver_type)
        else:
            raise NotImplementedError

    def forward(self, inputs, annotation=False, parallel=False):
        return self.ground_truth(inputs, annotation, parallel)
    
    def get_inputs(self, tour, first_explained_step, node_feats, dist_matrix=None):
        return self.ground_truth.get_inputs(tour, first_explained_step, node_feats, dist_matrix)
    
    def solve(self, step, input_tour, node_feats, instance_name=None):
        return self.ground_truth.solve(step, input_tour, node_feats, instance_name)