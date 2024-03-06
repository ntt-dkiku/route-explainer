from utils.data_utils.tsptw_dataset import TSPTWDataset
from utils.data_utils.pctsp_dataset import PCTSPDataset
from utils.data_utils.pctsptw_dataset import PCTSPTWDataset
from utils.data_utils.cvrp_dataset import CVRPDataset
from utils.data_utils.cvrptw_dataset import CVRPTWDataset
from utils.utils import save_dataset

def generate_dataset(num_samples, args):
    if args.problem == "tsptw":
        data_generator = TSPTWDataset(coord_dim=args.coord_dim,
                                      num_samples=num_samples,
                                      num_nodes=args.num_nodes,
                                      random_seed=args.random_seed,
                                      solver=args.solver, 
                                      classifier=args.classifier,
                                      annotation=args.annotation,
                                      parallel=args.parallel,
                                      num_cpus=args.num_cpus,
                                      distribution=args.distribution)
    elif args.problem == "pctsp":
        data_generator = PCTSPDataset(coord_dim=args.coord_dim,
                                      num_samples=num_samples,
                                      num_nodes=args.num_nodes,
                                      random_seed=args.random_seed,
                                      solver=args.solver, 
                                      classifier=args.classifier,
                                      annotation=args.annotation,
                                      parallel=args.parallel,
                                      num_cpus=args.num_cpus,
                                      penalty_factor=args.penalty_factor)
    elif args.problem == "pctsptw":
        data_generator = PCTSPTWDataset(coord_dim=args.coord_dim,
                                        num_samples=num_samples,
                                        num_nodes=args.num_nodes,
                                        random_seed=args.random_seed,
                                        solver=args.solver, 
                                        classifier=args.classifier,
                                        annotation=args.annotation,
                                        parallel=args.parallel,
                                        num_cpus=args.num_cpus,
                                        penalty_factor=args.penalty_factor)
    elif args.problem == "cvrp":
        data_generator = CVRPDataset(coord_dim=args.coord_dim,
                                     num_samples=num_samples,
                                     num_nodes=args.num_nodes,
                                     random_seed=args.random_seed,
                                     solver=args.solver, 
                                     classifier=args.classifier,
                                     annotation=args.annotation,
                                     parallel=args.parallel,
                                     num_cpus=args.num_cpus)
    elif args.problem == "cvrptw":
        data_generator = CVRPTWDataset(coord_dim=args.coord_dim,
                                       num_samples=num_samples,
                                       num_nodes=args.num_nodes,
                                       random_seed=args.random_seed,
                                       solver=args.solver, 
                                       classifier=args.classifier,
                                       annotation=args.annotation,
                                       parallel=args.parallel,
                                       num_cpus=args.num_cpus)
    else:
        raise NotImplementedError

    return data_generator.generate_dataset()

if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    parser = argparse.ArgumentParser(description='')
    # common settings
    parser.add_argument("--problem",     type=str, default="tsptw")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--data_type",   type=str, nargs="*", default=["all"], help="data type: 'all' or combo. of ['train', 'valid', 'test'].")
    parser.add_argument("--num_samples", type=int, nargs="*", default=[1000, 100, 100])
    parser.add_argument("--num_nodes",   type=int, default=20)
    parser.add_argument("--coord_dim",   type=int, default=2, help="only coord_dim=2 is supported for now.")
    parser.add_argument("--solver",      type=str, default="ortools", help="solver that outputs a tour")
    parser.add_argument("--classifier",  type=str, default="ortools", help="classifier for annotation")
    parser.add_argument("--annotation",  action="store_true")
    parser.add_argument("--parallel",    action="store_true")
    parser.add_argument("--num_cpus",    type=int, default=os.cpu_count())
    parser.add_argument("--output_dir",  type=str, default="data")
    # for TSPTW
    parser.add_argument("--distribution", type=str, default="da_silva")
    # for PCTSP
    parser.add_argument("--penalty_factor", type=float, default=3.)
    args = parser.parse_args()

    # 3d problems are not supported
    assert args.coord_dim == 2, "only coord_dim=2 is supported for now."

    # calc num. of total samples (train + valid + test samples)
    if args.data_type[0] == "all":
        assert len(args.num_samples) == 3, "please specify # samples for each of the three types (train/valid/test) when you set data_type 'all'. (e.g., --num_samples 1280000 1000 1000)"
    else:
        assert len(args.data_type) == len(args.num_samples), "please match # data_types and # elements in num_samples-arg"
    num_samples = np.sum(args.num_samples)

    # generate a dataset
    dataset = generate_dataset(num_samples, args)

    # split the dataset
    if args.data_type[0] == "all":
        types = ["train", "valid", "eval"]
    else:
        types = args.data_type
    num_sample_list = args.num_samples
    num_sample_list.insert(0, 0)
    start = 0
    for i, type_name in enumerate(types):
        start += num_sample_list[i]
        end = start + num_sample_list[i+1]
        divided_datset = dataset[start:end]
        output_fname = f"{args.output_dir}/{args.problem}/{type_name}_{args.problem}_{args.num_nodes}nodes_{num_sample_list[i+1]}samples_seed{args.random_seed}.pkl"
        save_dataset(divided_datset, output_fname)