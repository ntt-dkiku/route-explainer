import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPDecoder(nn.Module):
    def __init__(self, emb_dim, num_mlp_layers, num_classes, dropout):
        super().__init__()
        self.num_mlp_layers = num_mlp_layers

        # Decoder (MLP)
        self.mlp = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.mlp.append(nn.Linear(emb_dim, emb_dim, bias=True))
        self.mlp.append(nn.Linear(emb_dim, num_classes, bias=True))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initializing weights
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, graph_emb):
        """
        Paramters
        ---------
        graph_emb: torch.tensor [batch_size x emb_dim]

        Returns
        -------
        probs: torch.tensor [batch_size x num_classes]
            probabilities of classes
        """
        #----------
        # Decoding
        #----------
        h = graph_emb
        for i in range(self.num_mlp_layers):
            h = self.dropout(h)
            h = torch.relu(self.mlp[i](h))
        h = self.dropout(h)
        logits = self.mlp[-1](h)
        probs = F.log_softmax(logits, dim=-1)
        return probs