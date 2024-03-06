import torch.nn as nn
import numpy as np

class CFTourGenerator(nn.Module):
    def __init__(self, cf_solver):
        super().__init__()
        self.solver = cf_solver
        self.problem = cf_solver.problem

    def forward(self, factual_tour, vehicle_id, cf_step, cf_next_node_id, node_feats, dist_matrix=None):
        """
        solve an input instance with visited edges fixed

        Parameters
        ----------
        factual_tour: list [seq_length]
        cf_step: int
        cf_next_node_id: int
        node_feats: 

        Returns
        -------
        cf_tour: np.array [seq_length]
        """
        fixed_paths = self.get_fixed_paths(factual_tour, vehicle_id, cf_step, cf_next_node_id)
        cf_tours = self.solver.solve(node_feats, fixed_paths, dist_matrix=dist_matrix)
        if cf_tours is None:
            return
        if (cf_step > 0):
            for vehicle_id, cf_tour in enumerate(cf_tours):
                if cf_next_node_id in cf_tour:
                    if cf_step == 1:
                        if cf_tour[1] != cf_next_node_id:
                            cf_tours[vehicle_id] = np.flipud(cf_tour)
                        break
                    else:
                        if (factual_tour[vehicle_id][1] != cf_tour[1]):
                            cf_tours[vehicle_id] = np.flipud(cf_tour) # make direction of the cf tour the same as factual one
                        break
        print("aaaa", cf_tours)
        return cf_tours

    def get_fixed_paths(self, factual_tour, vehicle_id, cf_step, cf_next_node_id):
        visited_paths = np.append(factual_tour[vehicle_id][:cf_step], cf_next_node_id)
        return visited_paths

    # def get_avail_edges(self, factual_tour, cf_step, cf_next_node_id):
    #     visited_paths = np.append(factual_tour[:cf_step], cf_next_node_id)
    #     avail_edges = []
    #     # add fixed edges
    #     for i in range(len(visited_paths) - 1):
    #         avail_edges.append([visited_paths[i], visited_paths[i + 1]])
    #     print(avail_edges)

    #     # add rest avaialbel edges
    #     num_nodes = np.max(factual_tour) + 1
    #     visited = np.array([0] * num_nodes)
    #     for id in visited_paths:
    #         visited[id] = 1
    #     visited[factual_tour[0]] = 0
    #     visited[cf_next_node_id] = 0
    #     mask = visited < 1
    #     node_id = np.arange(num_nodes)
    #     feasible_node_id = node_id[mask]
    #     for j in range(len(feasible_node_id) - 1):
    #         for i in range(j + 1, len(feasible_node_id)):
    #             if ((feasible_node_id[j] == factual_tour[0]) and (feasible_node_id[i] == cf_next_node_id)) or ((feasible_node_id[i] == factual_tour[0]) and (feasible_node_id[j] == cf_next_node_id)):
    #                 continue
    #             avail_edges.append([feasible_node_id[j], feasible_node_id[i]])
    #     return np.array(avail_edges)

#-----------
# unit test
#-----------
if __name__ == "__main__":
    import argparse
    import random
    import matplotlib.pyplot as plt
    # FYI: 
    # - https://yu-nix.com/archives/python-path-get/
    # - https://www.delftstack.com/ja/howto/python/python-get-parent-directory/
    # - https://stackoverflow.com/questions/2817264/how-to-get-the-parent-dir-location
    import os
    import sys
    CURR_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
    sys.path.append(PARENT_DIR)
    from utils.util_vis import visualize_factual_and_cf_tours
    from lkh.lkh import LKH
    from models.ortools.ortools import ORTools
    from data_generator.tsptw.tsptw_dataset import generate_tsptw_instance

    parser = argparse.ArgumentParser(description='')
    # general settings
    parser.add_argument("--problem", type=str, default="tsptw")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--coord_dim", type=int, default=2)
    # LKH settings
    parser.add_argument("--max_trials", type=int, default=1000)
    parser.add_argument("--lkh_dir", type=str, default="lkh", help="Path to the binary of LKH")
    parser.add_argument("--io_dir", type=str, default="lkh_io_files")
    args = parser.parse_args()

    # models
    # cf_solver = LKH(args.problem, args.max_trials, args.random_seed, lkh_dir=args.lkh_dir, io_dir=args.io_dir)
    cf_solver = ORTools(args.problem)
    cf_generator = CFTourGenerator(cf_solver)

    # dataset
    if args.problem == "tsp":
        np.random.seed(args.random_seed)
        node_feats = np.random.uniform(size=[args.num_samples, args.num_nodes, args.coord_dim])
    elif args.problem == "tsptw":
        coords, time_window, grid_size = generate_tsptw_instance(num_nodes=args.num_nodes, grid_size=100, max_tw_gap=10, max_tw_size=1000, is_integer_instance=True, da_silva_style=True)
        node_feats = np.concatenate([coords, time_window], -1)
        node_feats = node_feats[None, :, :]

    # function ot automatically generate couterfactual visit 
    def get_random_cf_visit(factual_tour, random_seed=1234):
        # random.seed(random_seed)
        num_nodes = np.max(factual_tour) + 1
        step = random.randrange(len(factual_tour) - 2) # remove the last step (returning to the start-point)
        visited = np.array([0] * num_nodes)
        for i in range(step+1):
            visited[factual_tour[i]] = 1
        mask = visited < 1
        node_id = np.arange(num_nodes)
        feasible_node_id = node_id[mask]
        cf_next_id = random.choice(feasible_node_id) # select counterfactual
        return step, cf_next_id

    for i in range(len(node_feats)):
        # obtain a factual tour
        factual_tour = cf_solver.solve(node_feats[i])

        # counterfactual visit
        cf_step, cf_next_node_id = get_random_cf_visit(factual_tour, random_seed=args.random_seed)

        print(cf_step, cf_next_node_id)
        # obtain a counterfactual tour
        cf_tour = cf_generator(factual_tour, cf_step, cf_next_node_id, node_feats[i])

        print(factual_tour)
        print(cf_tour)

        # visualize the factual and counterfactual tours
        if args.problem == "tsp":
            coords = node_feats[i]
        elif args.problem == "tsptw":
            coord_dim = 2
            coords = node_feats[i, :, :coord_dim]
        visualize_factual_and_cf_tours(factual_tour, cf_tour, coords, cf_step, f"test{i}.png")
        break