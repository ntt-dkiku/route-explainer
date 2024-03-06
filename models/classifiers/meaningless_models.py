import torch
import torch.nn as nn

class RandomPredictor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: int or dict
            batch_size or dict of input features

        Returns
        -------
        probs: torch.tensor [batch_size x num_classes]
        """
        batch_size =  inputs if isinstance(inputs, int) else inputs["curr_node_id"].size(0)
        ranom_index = torch.randint(self.num_classes, (batch_size, self.num_classes))
        probs = torch.zeros(batch_size, self.num_classes).to(torch.float)
        probs.scatter_(-1, ranom_index, 1.0)
        return probs

    def get_inputs(self, tour, first_explained_step, node_feats):
        return len(tour[first_explained_step:-1])

class FixedClassPredictor(nn.Module):
    def __init__(self, predicted_class, num_classes):
        """
        Paramters
        ---------
        predicted_class: int
            a class that this predictor always predicts
        num_classes: int
            number of classes 
        """
        super().__init__()
        self.predicted_class = predicted_class
        self.num_classes = num_classes
        assert predicted_class < num_classes, f"predicted_class should be 0 - {num_classes}."

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: int or dict
            batch_size or dict of input features

        Returns
        -------
        probs: torch.tensor [batch_size x num_classes]
        """
        batch_size =  inputs if isinstance(inputs, int) else inputs["curr_node_id"].size(0)
        index = torch.full((batch_size, self.num_classes), self.predicted_class)
        probs = torch.zeros(batch_size, self.num_classes).to(torch.float)
        probs.scatter_(-1, index, 1.0)
        return probs
    
    def get_inputs(self, tour, first_explained_step, node_feats):
        return len(tour[first_explained_step:-1])