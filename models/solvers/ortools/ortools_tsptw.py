import numpy as np
from models.solvers.ortools.ortools_base import ORToolsBase

class ORToolsTSPTW(ORToolsBase):
    def __init__(self, large_value=1e+6, scaling=False):
        super().__init__(large_value, scaling)
    
    def scaling_feats(self, node_feats):
        return {
            key: (node_feat * self.large_value).astype(np.int64)
            if key in ("coords", "time_window") else
            node_feat
            for key, node_feat in node_feats.items()
        }
    
    def add_constraints(self, routing, transit_callback_index, manager, data, node_feats):
        """
        Adding time-window contraints

        Paramters
        ---------
        node_feats: dict
        """
        time_window = node_feats["time_window"]
        end_time = time_window[0, 1].item()
        routing.AddDimension(
            transit_callback_index,
            end_time, # max_wait_time (No limit)
            end_time, # end_time
            False,
            "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # penalty
        for i in range(len(data['distance_matrix'])):
            index = manager.NodeToIndex(i)
            routing.AddDisjunction([index], 100000000)

        # set time window
        for i in range(len(data['distance_matrix'])):
            index = manager.NodeToIndex(i)
            start = time_window[i, 0]
            end   = time_window[i, 1]
            time_dimension.CumulVar(index).SetRange(int(start), int(end))