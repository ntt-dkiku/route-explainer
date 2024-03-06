import os
import numpy as np
from utils.data_utils.dataset_base import DatasetBase
from models.solvers.general_solver import GeneralSolver
from models.classifiers.ground_truth.ground_truth_cvrptw import GroundTruthCVRPTW

class CVRPTWDataset(DatasetBase):
    """
    CVPRTW dataset by J.K. Falkner et al. https://arxiv.org/abs/2006.09100
    L. Xin et al. also adopt the dataset (normalized coords ver.). https://arxiv.org/abs/2110.07983
    """
    def __init__(self, coord_dim, num_samples, num_nodes, solver="ortools", classifier="ortools", annotation=True, parallel=True, random_seed=1234, num_cpus=os.cpu_count()):
        super().__init__(coord_dim, num_samples, num_nodes, annotation, parallel, random_seed, num_cpus)
        # CAPACITY = {
        #     20: 500,
        #     50: 750,
        #     100: 1000
        # }
        CAPACITY = {
            10: 20,
            20: 30,
            50: 40,
            100: 50
        }
        self.capacity = CAPACITY[num_nodes]
        problem = "cvrptw"
        solver_type = solver
        classifier_type = classifier
        self.cvrptw_solver = GeneralSolver(problem=problem, solver_type=solver_type)
        self.classifier = GroundTruthCVRPTW(solver_type=classifier_type)

    # @override
    def generate_instance(self, seed):
        np.random.seed(seed)
        #-------------
        # coordinates
        #-------------
        coords = np.random.uniform(size=(self.num_nodes+1, self.coord_dim))
        
        #---------
        # demands
        #---------
        # demands = np.random.normal(15, 10, (self.num_nodes+1, )).astype("int")
        # demands = np.maximum(np.minimum(np.ceil(np.abs(demands)), 42), 1) # clipping
        demands = np.random.randint(1, 10, size=(self.num_nodes+1, ))
        demands[0] = 0

        #-------------
        # time window
        #-------------
        dist = np.sqrt(((coords[0:1] - coords) ** 2).sum(-1)) * 100
        # define sampling horizon
        a0 = 0; b0 = 1000
        a_sample = np.floor(dist) + 1
        b_sample = b0 - a_sample - 10
        # sample horizon of each node
        a = np.random.uniform(size=(self.num_nodes+1,))
        a = (a * (b_sample - a_sample) + a_sample).astype("int")
        eps = np.maximum(np.abs(np.random.normal(0, 1, (self.num_nodes+1,))), 0.01)
        b = np.minimum(np.ceil(a + 300 * eps), b_sample)
        a[0] = a0; b[0] = b0
        a = a / 100
        b = b / 100
        time_window = np.concatenate((a[:, None], b[:, None]), -1)
        return {
            "coords": coords,
            "demand": demands.astype(np.int64),
            "time_window": time_window,
            "grid_size": np.array([1.0]),
            "capacity": np.array([self.capacity], dtype=np.int64)
        }

    # @override
    def annotate(self, instance):
        # Solve CVRPTW
        node_feats = instance
        cvrptw_tours = self.cvrptw_solver.solve(node_feats)
        if cvrptw_tours is None:
            return
        inputs = self.classifier.get_inputs(cvrptw_tours, 0, node_feats)
        labels = self.classifier(inputs, annotation=True)
        instance.update({"tour": cvrptw_tours, "labels": labels}) 
        return instance