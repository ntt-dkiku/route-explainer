import torch
import torch.nn as nn
import argparse
import json
import os
from models.classifiers.predictor import DecisionPredictor
from models.classifiers.meaningless_models import FixedClassPredictor, RandomPredictor
from models.classifiers.rule_based_models import kNearestPredictor
from models.classifiers.ground_truth.ground_truth import GroundTruth

class GeneralClassifier(nn.Module):
    def __init__(self, problem, model_type):
        super().__init__()
        self.model_type = model_type
        self.problem = problem
        self.model = self.get_model(problem, model_type)

    def change_model(self, problem, model_type):
        if self.model_type != model_type or self.problem != problem:
            self.model_type = model_type
            self.problem = problem
            self.model = self.get_model(problem, model_type)

    def get_model(self, problem, model_type):
        if model_type == "gnn":
            model_path = "checkpoints/model_20230309_101058/model_epoch4.pth"
            params = argparse.ArgumentParser()
            model_dir = os.path.split(model_path)[0]
            with open(f"{model_dir}/cmd_args.dat", "r") as f:
                params.__dict__ = json.load(f)
            model = DecisionPredictor(params.problem,
                                      params.emb_dim,
                                      params.num_mlp_layers,
                                      params.num_classes,
                                      params.dropout)
            model.load_state_dict(torch.load(model_path))
            return model
        elif model_type == "gt(ortools)":
            return GroundTruth(problem, solver_type="ortools")
        elif model_type == "gt(lkh)":
            return GroundTruth(problem, solver_type="lkh")
        elif model_type == "gt(concorde)":
            return GroundTruth(problem, solver_type="concorde")
        elif model_type == "random":
            return RandomPredictor(num_classes=2)
        elif model_type == "fixed":
            predicted_class = 0
            return FixedClassPredictor(predicted_class=predicted_class, num_classes=2)
        elif model_type == "knn":
            k = 5
            k_type = "num"
            return kNearestPredictor(problem, k, k_type)
        else:
            assert False, f"Invalid model type: {model_type}"
    
    def get_inputs(self, tour, first_explained_step, node_feats, dist_matrix=None):
        return self.model.get_inputs(tour, first_explained_step, node_feats, dist_matrix)
    
    def forward(self, inputs):
        return self.model(inputs)