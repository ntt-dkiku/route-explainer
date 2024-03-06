import numpy as np
from models.solvers.ortools.ortools_base import ORToolsBase

class ORToolsPCTSPTW(ORToolsBase):
    def __init__(self, large_value=1e+6, scaling=False):
        super().__init__(large_value, scaling)

    def scaling_feats(self, node_feats):
        return {
            key: (node_feat * self.large_value + 0.5).astype(np.int64)
            if key in ("coords", "prizes", "penalties", "time_window", "min_prize") else
            node_feat
            for key, node_feat in node_feats.items()
        }

    def add_constraints(self, routing, transit_callback_index, manager, data, node_feats):
        # Add penalties to nodes except for the depot
        # ORTools can ignore the nodes with taking the penalties
        penalties = node_feats["penalties"]
        for i in range(1, len(data['distance_matrix'])):
            index = manager.NodeToIndex(i)
            routing.AddDisjunction([index], penalties[i].item())

        # Add other constraints
        self.add_prize_constraints(routing, data, node_feats)
        self.add_time_window_constraints(routing, transit_callback_index, manager, data, node_feats)
    
    def add_time_window_constraints(self, routing, transit_callback_index, manager, data, node_feats):
        time_window = node_feats["time_window"]
        end_time = time_window[0, 1].item()
        routing.AddDimension(
            transit_callback_index,
            end_time, # max_wait_time
            end_time, # end_time
            False,
            "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # set time window
        for i in range(len(data['distance_matrix'])):
            index = manager.NodeToIndex(i)
            start = time_window[i, 0]
            end   = time_window[i, 1]
            time_dimension.CumulVar(index).SetRange(int(start), int(end))

    def add_prize_constraints(self, routing, data, node_feats):
        # Add prize dimension
        dim_name = "Prize"
        prizes = node_feats["prizes"]
        def prize_callback(from_node, to_node):
            return prizes[from_node].item()
        prize_callback_index = routing.RegisterTransitCallback(prize_callback)
        routing.AddDimension(
            prize_callback_index,
            0,  # Null capacity slack
            np.sum(prizes).item(),  # Upper bound
            True,  # Start cumul to zero
            dim_name)

        # Minimum prize constraints
        capacity_dimension = routing.GetDimensionOrDie(dim_name)
        for vehicle_id in range(data["num_vehicles"]):  # Only single vehicle
            capacity_dimension.CumulVar(routing.End(vehicle_id)).RemoveInterval(0, node_feats["min_prize"].item())