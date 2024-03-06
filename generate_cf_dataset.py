import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from utils.utils import load_dataset, save_dataset
from models.classifiers.ground_truth.ground_truth_base import get_visited_mask, get_tw_mask, get_cap_mask
from models.classifiers.ground_truth.ground_truth import GroundTruth
from models.solvers.general_solver import GeneralSolver
from models.cf_generator import CFTourGenerator


class CFDatasetBase():
    def __init__(self, problem, cf_generator, classifier, base_dataset, num_samples, random_seed, parallel, num_cpus):
        self.problem = problem
        self.parallel = parallel
        self.num_cpus = num_cpus
        self.seed = random_seed
        self.cf_generator = CFTourGenerator(cf_solver=GeneralSolver(problem, cf_generator))
        self.classifier = GroundTruth(problem, classifier)
        self.node_mask = NodeMask(problem)
        self.dataset = load_dataset(base_dataset)
        self.num_samples = len(self.dataset) if num_samples is None else num_samples

    def generate_cf_dataset(self):
        random.seed(self.seed)
        cf_dataset = []
        num_required_samples = self.num_samples
        end = False
        print("Data generation started.", flush=True)
        while(not end):
            dataset = self.dataset[:num_required_samples]
            self.dataset = np.roll(self.dataset, -num_required_samples)
            if self.parallel:
                instances = self.generate_labeldata_para(dataset, self.num_cpus)
            else:
                instances = self.generate_labeldata(dataset)
            cf_dataset.extend(filter(None, instances))
            num_required_samples = self.num_samples - len(cf_dataset) 
            if num_required_samples == 0:
                end = True
            else:
                print(f"No feasible tour was not found in {num_required_samples} instances. Trying other {num_required_samples} instances.", flush=True)
        print("Data generation completed.", flush=True)
        return cf_dataset

    def generate_labeldata(self, dataset):
        return [self.annotate(instance) for instance in tqdm(dataset, desc="Annotating instances")]

    def generate_labeldata_para(self, dataset, num_cpus):
        with Pool(num_cpus) as pool:
            annotation_data = list(tqdm(pool.imap(self.annotate, [instance for instance in dataset]), total=len(dataset), desc="Annotating instances"))
        return annotation_data

    def annotate(self, instance):
        # generate a counterfactual route randomly
        routes = instance["tour"]
        vehicle_id = random.randint(0, len(routes) - 1)
        if len(routes[vehicle_id]) - 2 <= 2:
            return
        cf_step    = random.randint(2, len(routes[vehicle_id]) - 2)
        route = routes[vehicle_id]
        mask = self.node_mask.get_mask(route, cf_step, instance)
        node_id = np.arange(len(instance["coords"]))
        feasible_node_id = node_id[mask]
        feasible_node_id = feasible_node_id[feasible_node_id != route[cf_step]].tolist()
        if len(feasible_node_id) == 0:
            return
        cf_visit  = random.choice(feasible_node_id)
        cf_routes = self.cf_generator(routes, vehicle_id, cf_step, cf_visit, instance)
        if cf_routes is None:
            return

        # annotate each edge
        inputs = self.classifier.get_inputs(cf_routes, 0, instance)
        labels = self.classifier(inputs, annotation=True)

        # update tours and lables
        instance["tour"] = cf_routes
        instance["labels"] = labels
        return instance

class NodeMask():
    def __init__(self, problem):
        self.problem = problem

        if self.problem == "tsptw":
            self.mask_func = get_tsptw_mask
        elif self.problem == "pctsp":
            self.mask_func = get_pctsp_mask
        elif self.problem == "pctsptw":
            self.mask_func = get_pctsptw_mask
        elif self.problem == "cvrp":
            self.mask_func = get_cvrp_mask
        else:
            NotImplementedError

    def get_mask(self, route, step, instance):
        return self.mask_func(route, step, instance)

def get_tsptw_mask(route, step, instance):
    visited = get_visited_mask(route, step, instance)
    not_exceed_tw = get_tw_mask(route, step, instance)
    return ~visited & not_exceed_tw

def get_pctsp_mask(route, step, instance):
    visited = get_visited_mask(route, step, instance)
    return ~visited

def get_pctsptw_mask(route, step, instance):
    visited = get_visited_mask(route, step, instance)
    not_exceed_tw = get_tw_mask(route, step, instance)
    return ~visited & not_exceed_tw

def get_cvrp_mask(route, step, instance):
    visited = get_visited_mask(route, step, instance)
    less_than_cap = get_cap_mask(route, step, instance)
    return ~visited & less_than_cap

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--problem",      type=str, default="tsptw")
    parser.add_argument("--base_dataset", type=str, required=True)
    parser.add_argument("--cf_generator", type=str, default="ortools")
    parser.add_argument("--classifier",   type=str, default="ortools")
    parser.add_argument("--num_samples",  type=int, default=None)
    parser.add_argument("--random_seed",  type=int, default=1234)
    parser.add_argument("--parallel",     action="store_true")
    parser.add_argument("--num_cpus",     type=int, default=4)
    parser.add_argument("--output_dir",   type=str, default="data")
    args = parser.parse_args()

    dataset_gen = CFDatasetBase(args.problem,
                                args.cf_generator,
                                args.classifier,
                                args.base_dataset,
                                args.num_samples,
                                args.random_seed,
                                args.parallel,
                                args.num_cpus)
    cf_dataset = dataset_gen.generate_cf_dataset()

    output_fname = f"{args.output_dir}/{args.problem}/cf_{dataset_gen.num_samples}samples_seed{args.random_seed}_base_{os.path.basename(args.base_dataset)}.pkl"
    save_dataset(cf_dataset, output_fname)