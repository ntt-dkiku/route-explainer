import numpy as np
from scipy.spatial import distance
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class ORToolsBase():
    def __init__(self, large_value=1e+6, scaling=False):
        self.coord_dim = 2
        self.large_value = large_value
        self.scaling = scaling

    def preprocess_data(self, node_feats, dist_matrix=None):
        if self.scaling:
            node_feats = self.scaling_feats(node_feats)
        data = {}
        # convert set of corrdinates to a distance matrix
        if dist_matrix is None:
            coords = node_feats["coords"]
            _dist_matrix = distance.cdist(coords, coords, 'euclidean').astype(np.int64)
        else:
            _dist_matrix = dist_matrix
            if self.scaling:
                _dist_matrix *= self.large_value
        data['distance_matrix'] = _dist_matrix.tolist()
        data['num_vehicles'] = 1
        data['depot'] = 0
        if "service_time" in node_feats:
            data["service_time"] = node_feats["service_time"].astype(np.int64).tolist()
        else:
            data["service_time"] = np.zeros(len(data['distance_matrix']), dtype=np.int64).tolist()
        return node_feats, data

    def scaling_feats(self, node_feats):
        raise NotImplementedError
    
    def solve(self, node_feats, fixed_path=None, dist_matrix=None, instance_name=None):
        """
        Paramters
        ---------
        node_feats: np.array [num_nodes x node_dim]
            the first (coord_dim) dims are coord_dim
        fixed_path: 

        Returns
        -------
        tour: np.array [seq_length]
        """
        node_feats, data = self.preprocess_data(node_feats, dist_matrix)
        
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']), data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node] + data["service_time"][from_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        self.add_constraints(routing, transit_callback_index, manager, data, node_feats)

        # fix a partial path
        if fixed_path is not None:
            fixed_path = self.index2node(fixed_path, manager)
            print(fixed_path)
            routing.CloseModel() # <- very important. this should be called at last
            # ApplyLocks supports only single vehicle
            # routing.ApplyLocks(fixed_path)
            # As ApplyLocksToAllVehicles does not contain depot,
            # remove depot id from fixed_path (first element of fixed_path is alwawys depot_id)
            routing.ApplyLocksToAllVehicles([fixed_path[1:]], False)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 5# int(1 * 60) #120

        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment is None:
            return 
        # assert assignment is not None, "Found no solution."
        return self.get_tour_list(data, manager, routing, assignment)
    
    def index2node(self, path, manager):
        mod_path = []
        for i in range(len(path)):
            mod_path.append(manager.IndexToNode(path[i].item(0)))
        return mod_path

    def get_tour_list(self, data, manager, routing, solution):
        """
        Returns
        -------
        tour: 2d list [num_vehicles x seq_length]
        """
        num_vehicles = data["num_vehicles"]
        tour = [[] for _ in range(num_vehicles)]
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                tour[vehicle_id].append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            tour[vehicle_id].append(manager.IndexToNode(index))
        # remove unused vehicles
        tour = [vehicle_tour for vehicle_tour in tour if len(vehicle_tour) > 2]
        return tour

    def add_constraints(self, routing, transit_callback_index, manager, data, node_feats):
        pass