import numpy as np
from scipy.spatial import distance
from models.solvers.ortools.ortools_base import ORToolsBase

class ORToolsCVRP(ORToolsBase):
    def __init__(self, large_value=1e+6, scaling=False):
        super().__init__(large_value, scaling)
    
    # @override
    def preprocess_data(self, node_feats):
        if self.scaling:
            node_feats = self.scaling_feats(node_feats)
        coords = node_feats["coords"]
        demands = node_feats["demand"]
        capacity = node_feats["capacity"]

        data = {}
        # convert set of corrdinates to a distance matrix
        dist_matrix = distance.cdist(coords, coords, "euclidean").round().astype(np.int64)
        data["distance_matrix"] = dist_matrix.tolist()
        data["num_vehicles"] = 10
        data["depot"] = 0
        data["demands"] = demands.tolist()
        data["vehicle_capacities"] = capacity.tolist() * data["num_vehicles"]
        return node_feats, data

    # @override
    def scaling_feats(self, node_feats):
        return {
            key: (node_feat * self.large_value).astype(np.int64) 
                 if key == "coords" else 
                 node_feat
            for key, node_feat in node_feats.items()
        }
    
    # @override
    def add_constraints(self, routing, transit_callback_index, manager, data, node_feats):
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity"
        )