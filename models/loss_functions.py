import torch
import torch.nn as nn
from utils.utils import batched_bincount
import torch.nn.functional as F


class GeneralCrossEntropy(nn.Module):
    def __init__(self, weight_type: str, beta : float = 0.99, is_sequential: bool = True):
        super().__init__()
        self.weight_type = weight_type
        self.beta = beta
        if weight_type == "seq_cbce":
            assert is_sequential == True
            self.loss_func = SeqCBCrossEntropy(beta=beta)
        elif weight_type == "cbce":
            self.loss_func = CBCrossEntropy(beta=beta, is_sequential=is_sequential)
        elif weight_type == "wce":
            self.loss_func = WeightedCrossEntropy(is_sequential=is_sequential)
        elif weight_type == "ce":
            self.loss_func = CrossEntropy(is_sequential=is_sequential)
        else:
            NotImplementedError

    def forward(self, 
                preds: torch.Tensor, 
                labels: torch.Tensor,
                pad_mask: torch.Tensor = None):
        return self.loss_func(preds, labels, pad_mask)


class SeqCBCrossEntropy(nn.Module):
    def __init__(self, beta : float = 0.99):
        super().__init__()
        self.beta = beta

    def forward(self,
                preds: torch.Tensor, 
                labels: torch.Tensor,
                pad_mask: torch.Tensor):
        """
        Sequential Class-alanced Cross Entropy Loss (Our proposal)

        Parameters
        -----------
        preds: torch.Tensor [batch_size, max_seq_length, num_classes]
        labels: torch.Tensor [batch_size, max_seq_length]
        pad_mask: torch.Tensor [batch_size, max_seq_length]

        Returns
        -------
        loss: torch.Tensor [1]
        """
        seq_length_batch = pad_mask.sum(-1) # [batch_size]
        seq_length_list = torch.unique(seq_length_batch) # [num_unique_seq_length]
        batch_size = preds.size(0)
        loss = 0
        for seq_length in seq_length_list:
            extracted_batch = (seq_length_batch == seq_length) # [batch_size]
            extracted_preds = preds[extracted_batch]   # [num_extracted_batch]
            extracted_labels = labels[extracted_batch] # [num_extracted_batch]
            extracted_batch_size = extracted_labels.size(0)
            bin = batched_bincount(extracted_labels.T, 1, extracted_preds.size(-1)) # [seq_length x num_classes]
            weight = (1 - self.beta) / (1 - self.beta**bin + 1e-8)
            for seq_no in range(seq_length.item()):
                loss += (extracted_batch_size / batch_size) * F.nll_loss(extracted_preds[:, seq_no], extracted_labels[:, seq_no], weight=weight[seq_no])
        return loss

class CBCrossEntropy(nn.Module):
    def __init__(self, beta : float = 0.99, is_sequential: bool = True):
        super().__init__()
        self.beta = beta
        self.is_sequential = is_sequential

    def forward(self,
                preds: torch.Tensor, 
                labels: torch.Tensor,
                pad_mask: torch.Tensor = None):
        if self.is_sequential:
            mask = pad_mask.view(-1)
            preds = preds.view(-1, preds.size(-1))
            bin = labels.view(-1)[mask].bincount()
            weight = (1 - self.beta) / (1 - self.beta**bin + 1e-8)
            loss = F.nll_loss(preds[mask], labels.view(-1)[mask], weight=weight)
        else:
            bincount = labels.view(-1).bincount()
            weight = (1 - self.beta) / (1 - self.beta**bincount + 1e-8)
            loss = F.nll_loss(preds, labels.squeeze(-1), weight=weight)
        return loss

class WeightedCrossEntropy(nn.Module):
    def __init__(self, is_sequential: bool = True, norm: str = "min"):
        super().__init__()
        self.is_sequential = is_sequential
        if norm == "min":
            self.norm = torch.min
        elif norm == "max":
            self.norm = torch.max
    def forward(self,
                preds: torch.Tensor, 
                labels: torch.Tensor,
                pad_mask: torch.Tensor = None):
        if self.is_sequential:
            mask = pad_mask.view(-1)
            preds = preds.view(-1, preds.size(-1))
            bin = labels.view(-1)[mask].bincount()
            weight = self.norm(bin) / (bin + 1e-8)
            loss = F.nll_loss(preds[mask], labels.view(-1)[mask], weight=weight)
        else:
            bincount = labels.view(-1).bincount()
            weight = self.norm(bin) / (bin + 1e-8)
            loss = F.nll_loss(preds, labels.squeeze(-1), weight=weight)
        return loss

class CrossEntropy(nn.Module):
    def __init__(self, is_sequential: bool = True):
        super().__init__()
        self.is_sequential = is_sequential

    def forward(self,
                preds: torch.Tensor, 
                labels: torch.Tensor,
                pad_mask: torch.Tensor = None):
        if self.is_sequential:
            mask = pad_mask.view(-1)
            preds = preds.view(-1, preds.size(-1))
            loss = F.nll_loss(preds[mask], labels.view(-1)[mask])
        else:
            loss = F.nll_loss(preds, labels.squeeze(-1))
        return loss