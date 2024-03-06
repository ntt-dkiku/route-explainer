import os
import argparse
import json
import multiprocessing
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from utils.util_calc import TemporalConfusionMatrix
from models.classifiers.nn_classifiers.nn_classifier import NNClassifier
from models.classifiers.ground_truth.ground_truth import GroundTruth
from models.classifiers.ground_truth.ground_truth_base import FAIL_FLAG
from utils.data_utils.tsptw_dataset import TSPTWDataloader
from utils.data_utils.pctsp_dataset import PCTSPDataloader
from utils.data_utils.pctsptw_dataset import PCTSPTWDataloader
from utils.data_utils.cvrp_dataset import CVRPDataloader
from utils.utils import set_device
from utils.utils import load_dataset

def load_eval_dataset(dataset_path, problem, model_type, batch_size, num_workers, parallel, num_cpus):
    if model_type == "nn":
        if problem == "tsptw":
            eval_dataset = TSPTWDataloader(dataset_path, sequential=True, parallel=parallel, num_cpus=num_cpus)
        elif problem == "pctsp":
            eval_dataset = PCTSPDataloader(dataset_path, sequential=True, parallel=parallel, num_cpus=num_cpus)
        elif problem == "pctsptw":
            eval_dataset = PCTSPTWDataloader(dataset_path, sequential=True, parallel=parallel, num_cpus=num_cpus)
        elif problem == "cvrp":
            eval_dataset = CVRPDataloader(dataset_path, sequential=True, parallel=parallel, num_cpus=num_cpus)
        else:
            raise NotImplementedError

        #------------
        # dataloader
        #------------
        def pad_seq_length(batch):
            data = {}
            for key in batch[0].keys():
                padding_value = True if key == "mask" else 0.0
                # post-padding
                data[key] = torch.nn.utils.rnn.pad_sequence([d[key] for d in batch], batch_first=True, padding_value=padding_value)
            pad_mask = torch.nn.utils.rnn.pad_sequence([torch.full((d["mask"].size(0), ), True) for d in batch], batch_first=True, padding_value=False)
            data.update({"pad_mask": pad_mask})
            return data
        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=pad_seq_length,
                                     num_workers=num_workers)
        return eval_dataloader
    else:
        eval_dataset = load_dataset(dataset_path)
        return eval_dataset

def eval_classifier(problem: str, 
                    dataset, 
                    model_type: str, 
                    model_dir: str = None, 
                    gpu: int = -1, 
                    num_workers: int = 4, 
                    batch_size: int = 128, 
                    parallel: bool = True,
                    solver: str = "ortools",
                    num_cpus: int = 1):
    #--------------
    # gpu settings
    #--------------
    use_cuda, device = set_device(gpu)

    #-------
    # model
    #-------
    num_classes = 3 if problem == "pctsptw" else 2
    if model_type == "nn":
        assert model_dir is not None, "please specify model_path when model_type is nn."
        params = argparse.ArgumentParser()
        # model_dir = os.path.split(args.model_path)[0]
        with open(f"{model_dir}/cmd_args.dat", "r") as f:
            params.__dict__ = json.load(f)
        assert params.problem == problem, "problem of the trained model should match that of the dataset"
        model = NNClassifier(problem=params.problem,
                             node_enc_type=params.node_enc_type,
                             edge_enc_type=params.edge_enc_type,
                             dec_type=params.dec_type,
                             emb_dim=params.emb_dim,
                             num_enc_mlp_layers=params.num_enc_mlp_layers,
                             num_dec_mlp_layers=params.num_dec_mlp_layers,
                             num_classes=num_classes,
                             dropout=params.dropout,
                             pos_encoder=params.pos_encoder)
        # load trained weights (the best epoch)
        with open(f"{model_dir}/best_epoch.dat", "r") as f:
            best_epoch = int(f.read())
        print(f"loaded {model_dir}/model_epoch{best_epoch}.pth.")
        model.load_state_dict(torch.load(f"{model_dir}/model_epoch{best_epoch}.pth"))
        if use_cuda:
            model.to(device)
        is_sequential = model.is_sequential
    elif model_type == "ground_truth":
        model = GroundTruth(problem=problem, solver_type=solver)
        is_sequential = False
    else:
        assert False, f"Invalid model type: {model_type}"

    #---------
    # Metrics
    #---------
    overall_accuracy = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    eval_accuracy_dict = {} # MulticlassAccuracy(num_classes=num_classes, average="macro")
    temp_confmat_dict  = {} # TemporalConfusionMatrix(num_classes=num_classes, seq_length=50, device=device)
    temporal_accuracy_dict  = {}
    num_nodes_dist_dict = {}
                        
    #------------
    # Evaluation
    #------------
    if model_type == "nn":
        model.eval()
        eval_time = 0.0
        print("Evaluating models ...", end="")
        start_time = time.perf_counter()
        for data in dataset:
            if use_cuda:
                data = {key: value.to(device) for key, value in data.items()}
            if not is_sequential:
                shp = data["curr_node_id"].size()
                data = {key: value.flatten(0, 1) for key, value in data.items()}
            probs = model(data) # [batch_size x num_classes] or [batch_size x max_seq_length x num_classes]
            if not is_sequential:
                probs = probs.view(*shp, -1) # [batch_size x max_seq_length x num_classes]
                data["labels"] = data["labels"].view(*shp)
                data["pad_mask"] = data["pad_mask"].view(*shp)
            #------------
            # evaluation
            #------------
            start_eval_time = time.perf_counter()
            # accuracy
            seq_length_list = torch.unique(data["pad_mask"].sum(-1)) 
            for seq_length_tensor in seq_length_list:
                seq_length = seq_length_tensor.item()
                if seq_length not in eval_accuracy_dict.keys():
                    eval_accuracy_dict[seq_length] = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
                    temp_confmat_dict[seq_length]  = TemporalConfusionMatrix(num_classes=num_classes, seq_length=seq_length, device=device)
                    temporal_accuracy_dict[seq_length] = [MulticlassF1Score(num_classes=num_classes, average="macro").to(device) for _ in range(seq_length)]
                    num_nodes_dist_dict[seq_length] = 0
                seq_length_mask = (data["pad_mask"].sum(-1) == seq_length) # [batch_size]
                extracted_labels = data["labels"][seq_length_mask]
                extracted_probs  = probs[seq_length_mask]
                extracted_mask   = data["pad_mask"][seq_length_mask].view(-1) # [batch_size x max_seq_length] -> [(batch_size*max_seq_length)]
                eval_accuracy_dict[seq_length](extracted_probs.argmax(-1).view(-1)[extracted_mask], extracted_labels.view(-1)[extracted_mask])
                mask = data["pad_mask"].view(-1)
                overall_accuracy(probs.argmax(-1).view(-1)[mask], data["labels"].view(-1)[mask])
                # confusion matrix
                temp_confmat_dict[seq_length].update(probs.argmax(-1), data["labels"], data["pad_mask"])         
                # temporal accuracy
                for step in range(seq_length):
                    temporal_accuracy_dict[seq_length][step](extracted_probs[:, step, :], extracted_labels[:, step])
                # number of samples whose sequence length is seq_length
                num_nodes_dist_dict[seq_length] += len(extracted_labels)
            eval_time += time.perf_counter() - start_eval_time
        calc_time = time.perf_counter() - start_time - eval_time
        total_eval_accuracy = {key: value.compute().item() for key, value in eval_accuracy_dict.items()}
        overall_accuracy = overall_accuracy.compute() #.item()
        temporal_confmat = {key: value.compute() for key, value in temp_confmat_dict.items()}
        temporal_accuracy = {key: [value.compute().item() for value in values] for key, values in temporal_accuracy_dict.items()}
        print("done")
        return overall_accuracy, total_eval_accuracy, temporal_accuracy, calc_time, temporal_confmat, num_nodes_dist_dict
    else:
        eval_accuracy = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        print("Loading data ...", end=" ")
        with multiprocessing.Pool(num_cpus) as pool:
            input_list = list(pool.starmap(model.get_inputs, [(instance["tour"], 0, instance) for instance in dataset]))
        print("done")

        print("Infering labels ...", end="")
        pool = multiprocessing.Pool(num_cpus)
        start_time = time.perf_counter()
        prob_list = list(pool.starmap(model, tqdm([(inputs, False, False) for inputs in input_list])))
        calc_time = time.perf_counter() - start_time
        pool.close()
        print("done")

        print("Evaluating models ...", end="")
        for i, instance in enumerate(dataset):
            labels = instance["labels"]
            for vehicle_id in range(len(labels)):
                for step, label in labels[vehicle_id]:
                    pred_label = prob_list[i][vehicle_id][step-1] # [num_classes]
                    if pred_label == FAIL_FLAG:
                        pred_label = label - 1 if label != 0 else label + 1
                    eval_accuracy(torch.LongTensor([pred_label]).view(1, -1), torch.LongTensor([label]).view(1, -1))
        total_eval_accuracy = eval_accuracy.compute()
        print("done")
        return total_eval_accuracy.item(), calc_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #-----------------
    # general settings
    #-----------------
    parser.add_argument("--gpu", default=-1, type=int, help="Used GPU Number: gpu=-1 indicates using cpu")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers in dataloader")
    parser.add_argument("--parallel", )

    #-------------
    # data setting
    #-------------
    parser.add_argument("--dataset_path", type=str, help="Path to a dataset", required=True)
    
    #------------------
    # Metrics settings
    #------------------


    #----------------
    # model settings
    #----------------
    parser.add_argument("--model_type", type=str, default="nn", help="Select from [nn, ground_truth]")
    # nn classifier
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--parallel", action="store_true")
    # ground truth
    parser.add_argument("--solver", type=str, default="ortools")
    parser.add_argument("--num_cpus", type=int, default=os.cpu_count())
    args = parser.parse_args()

    problem   = str(os.path.basename(os.path.dirname(args.dataset_path)))

    dataset = load_eval_dataset(args.dataset_path, problem, args.model_type, args.batch_size, args.num_workers, args.parallel, args.num_cpus)
    eval_classifier(problem=problem, 
         dataset=dataset,
         model_type=args.model_type,
         model_dir=args.model_dir,
         gpu=args.gpu,
         num_workers=args.num_workers,
         batch_size=args.batch_size,
         parallel=args.parallel,
         solver=args.solver,
         num_cpus=args.num_cpus)