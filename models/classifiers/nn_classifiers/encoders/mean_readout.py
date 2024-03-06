import math
import torch
import torch.nn as nn

class MeanReadout(nn.Module):
    def __init__(self, state_dim, emb_dim, dropout):
        super().__init__()
        self.state_dim = state_dim
        self.emb_dim   = emb_dim

        # initial embedding for state
        if state_dim > 0:
            self.init_linear_state = nn.Linear(state_dim, emb_dim)

        # out linear layer
        self.out_linear = nn.Linear((1 + int(state_dim > 0))*emb_dim, emb_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()
    
    def reset_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, inputs, node_emb):
        """
        Paramters
        ---------
        inputs: dict
            mask: torch.LongTensor [batch_size x num_nodes]
            state: torch.FloatTensor [batch_size x state_dim]
        node_emb: torch.tensor [batch_size x num_nodes x emb_dim]
            node embeddings obtained from the node encoder

        Returns
        -------
        h: torch.tensor [batch_size x emb_dim]
            graph embeddings
        """
        mask = inputs["mask"]
        state = inputs["state"]
        node_emb = self.dropout(node_emb)

        # pooling with a mask
        mask = mask.unsqueeze(-1).expand_as(node_emb)
        node_emb = node_emb * mask
        h = torch.mean(node_emb, dim=1) # [batch_size x emb_dim]
        
        # out linear layer
        if state is not None and self.state_dim > 0:
            state_emb = self.init_linear_state(state) # [batch_size x emb_dim]
            h  = torch.cat((h, state_emb), -1) # [batch_size x (2*emb_dim)]

        return self.out_linear(h)