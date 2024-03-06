import os
from tqdm import tqdm
import multiprocessing
import numpy as np
from utils.utils import load_dataset, calc_tour_length
from models.solvers.general_solver import GeneralSolver
from models.classifiers.ground_truth.ground_truth import GroundTruth

def eval_solver(solver, instance):
    tour = solver.solve(instance)
    tour_length = calc_tour_length(tour[0], instance["coords"])
    return tour_length

def eval(data_path, problem, solver_name, fix_edges, parallel):
    dataset = load_dataset(data_path)
    num_cpus = os.cpu_count() if parallel else 1
    if fix_edges:
        solver = GroundTruth(problem, solver_name)
        if parallel:
            with multiprocessing.Pool(num_cpus) as pool:
                tours = list(tqdm(pool.starmap(solver.solve, [(step, instance["tour"][vehicle_id], instance, f"{i}-{vehicle_id}-{step}")
                                                              for i, instance in enumerate(dataset)
                                                              for vehicle_id in range(len(instance["tour"]))
                                                              for step in range(1, len(instance["tour"][vehicle_id]))]), desc=f"Solving {data_path} with {solver_name}"))
        else:
            tours = []
            for i, instance in enumerate(dataset):
                for vehicle_id in range(len(instance["tour"])):
                    for step in range(1, len(instance["tour"][vehicle_id])):
                        tours.append(solver.solve(step, instance["tour"][vehicle_id], instance, f"{i}-{vehicle_id}-{step}"))
        tour_length = {key: [] for key in tours[0].keys()}
        for tour in tours:
            for key, value in tour.items():
                tour_length[key].append(value)
    else:
        solver = GeneralSolver(problem, solver_name)
        with multiprocessing.Pool(num_cpus) as pool:
            tour_length = list(tqdm(pool.starmap(eval_solver, [(solver, instance) for instance in dataset]), total=len(dataset), desc="Solving instances"))

    feasible_ratio = 0.0
    penalty = 0.0
    avg_tour_length = np.mean(tour_length["tsp"])
    std_tour_length = np.std(tour_length["tsp"])
    return avg_tour_length, std_tour_length, feasible_ratio, penalty

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="tsptw", type=str, help="Problem type: [tsptw, pctsp, pctsptw, cvrp]")
    parser.add_argument("--solver_name", type=str, default="ortools", help="Select from ")
    parser.add_argument("--data_path", type=str, help="Path to a dataset", required=True)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fix_edges", action="store_true")
    args = parser.parse_args()

    avg_tour_length, std_tour_length, feasible_ratio, penalty = eval(data_path=args.data_path,
                                                                     problem=args.problem,
                                                                     solver_name=args.solver_name,
                                                                     fix_edges=args.fix_edges,
                                                                     parallel=args.parallel)
    print(f"tour_length: {avg_tour_length} +/- {std_tour_length}")