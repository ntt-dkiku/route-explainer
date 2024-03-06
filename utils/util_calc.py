import numpy as np
import torch
from torchmetrics.classification import ConfusionMatrix

def calc_tour_length(tour, coords):
    tour_length = []
    for i in range(len(tour) - 1):
        path_length = np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
        tour_length.append(path_length)
    tour_length = np.sum(tour_length)
    return tour_length

class TemporalConfusionMatrix():
    def __init__(self, num_classes: int, seq_length: int, device: str):
        if num_classes == 2:
            task = "binary" 
        elif num_classes > 2:
            task = "multiclass"
        else:
            assert False, "Invalid num_classes. It should be more than 2"
        
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.temp_confmat = [ConfusionMatrix(task=task, num_classes=num_classes).to(device) for _ in range(seq_length)]

    def update(self, 
               preds: torch.Tensor,
               labels: torch.Tensor,
               mask: torch.Tensor):
        """
        Parameters
        ----------
        preds: predicted labels [batch_size x max_seq_length]
        label: ground truth labels [batch_size x max_seq_length]
        mask:  mask of padding [batch_size x max_seq_length]
        """
        batch_mask = (mask.sum(-1) == self.seq_length) # [batch_size]
        preds = preds[batch_mask]   # [fixed_batch_size x max_seq_length]
        labels = labels[batch_mask] # [fixed_batch_size x max_seq_length]
        for i in range(self.seq_length):
            self.temp_confmat[i](preds[:, i], labels[:, i])

    def compute(self, device="cpu"):
        return [confmat.compute().to(device) for confmat in self.temp_confmat]

def calc_route_length(routes: list, coords: np.array):
    route_length = []
    num_vehicles = get_num_vehicles(routes)
    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        for i in range(len(route) - 1):
            edge_length = np.linalg.norm(coords[route[i]] - coords[route[i + 1]])
            route_length.append(edge_length)
    route_length = np.sum(route_length)
    return route_length

def get_num_vehicles(routes: list):
    return len(routes)

def calc_class_ratio(labels: torch.Tensor, routes: list):
    """
    Parameters
    ----------
    lables: torch.Tensor [num_vehicles, max_seq_length]
    routes: 2d list [num_vehicles, seq_length]
    """
    if isinstance(labels, torch.Tensor):
        num_classes = 3 # labels.max().item() + 1
        class_list = [0 for _ in range(num_classes)]
        print(labels, routes)
        for vehicle_id in range(get_num_vehicles(routes)):
            for step in range(len(routes[vehicle_id])-1):
                class_list[labels[vehicle_id, step].item()] += 1
    else:
        if len(labels) > 1:
            num_classes = np.max(np.max(labels)) + 1
        else:    
            num_classes = np.max(labels) + 1
        class_list = [0 for _ in range(num_classes)]
        for vehicle_id in range(get_num_vehicles(routes)):
            for step in range(len(routes[vehicle_id])-1):
                class_list[labels[vehicle_id][step]] += 1
    class_list = np.array(class_list)
    return class_list / np.sum(class_list)

def calc_feasible_ratio(routes: list, coords: np.array):
    return len(np.unique(routes)) / len(coords)

def calc_total_prizes(routes: list, prizes: np.array):
    total_prizes = 0.
    num_vehicles = get_num_vehicles(routes)
    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        for i in range(len(route) - 1):
            total_prizes += prizes[route[i]]
    return total_prizes

def calc_total_penlties(routes: list, penalities: np.array):
    total_penalites = 0.
    num_nodes = len(penalities)
    node_ids = np.arange(num_nodes)
    visited_node_ids = np.unique(routes)
    unvisited_node_ids = np.setdiff1d(node_ids, visited_node_ids)
    for unvisited_node_id in unvisited_node_ids:
        total_penalites += penalities[unvisited_node_id]
    return total_penalites