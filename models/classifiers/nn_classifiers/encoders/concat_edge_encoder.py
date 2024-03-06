import math
import torch
import torch.nn as nn

class ConcatEdgeEncoder(nn.Module):
    def __init__(self, state_dim, emb_dim, dropout):
        super().__init__()
        self.state_dim = state_dim
        self.emb_dim   = emb_dim
        self.norm_factor = 1 / math.sqrt(emb_dim)

        # initial embedding for state
        if state_dim > 0:
            self.init_linear_state = nn.Linear(state_dim, emb_dim)

        # out linear layer
        self.out_linear = nn.Linear((2 + int(state_dim > 0)) * emb_dim, emb_dim)

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
            curr_node_id: torch.LongTensor [batch_size]
            next_node_id: torch.LongTensor [batch_size]
            mask: torch.LongTensor [batch_size x num_nodes]
            state: torch.FloatTensor [batch_size x state_dim]
        node_emb: torch.tensor [batch_size x num_nodes x emb_dim]
            node embeddings obtained from the node encoder

        Returns
        -------
        h: torch.tensor [batch_size x emb_dim]
            edge embeddings 
        """
        curr_node_id = inputs["curr_node_id"]
        next_node_id = inputs["next_node_id"]
        state = inputs["state"]
        batch_size = curr_node_id.size(0)

        #--------------------------------
        # generate queries, keys, values
        #--------------------------------
        node_emb = self.dropout(node_emb)
        curr_emb = node_emb.gather(1, curr_node_id[:, None, None].expand(batch_size, 1, self.emb_dim))
        next_emb = node_emb.gather(1, next_node_id[:, None, None].expand(batch_size, 1, self.emb_dim))
        if state is not None and self.state_dim > 0:
            state_emb = self.init_linear_state(state) # [batch_size x emb_dim]
            edge_emb = torch.cat((curr_emb, next_emb, state_emb[:, None, :]), -1) # [batch_size x 1 x (3*emb_dim)]
        else:
            edge_emb = torch.cat((curr_emb, next_emb), -1) # [batch_size x 1 x (2*emb_dim)]
        edge_emb = edge_emb.squeeze(1) # [batch_size x (2*emb_dim)]
        return self.out_linear(edge_emb)