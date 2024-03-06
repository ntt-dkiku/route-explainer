import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionGraphEncoder(nn.Module):
    def __init__(self, coord_dim, node_dim, state_dim, emb_dim, dropout):
        super().__init__()
        self.coord_dim = coord_dim
        self.node_dim  = node_dim 
        self.emb_dim   = emb_dim
        self.state_dim = state_dim
        self.norm_factor = 1 / math.sqrt(emb_dim)

        # initial embedding
        self.init_linear_node  = nn.Linear(node_dim, emb_dim)
        self.init_linear_depot = nn.Linear(coord_dim, emb_dim)
        if state_dim > 0:
            self.init_linear_state = nn.Linear(state_dim, emb_dim)
        
        # An attention layer
        self.w_q = nn.Parameter(torch.FloatTensor((2 + int(state_dim > 0)) * emb_dim, emb_dim))
        self.w_k = nn.Parameter(torch.FloatTensor(2 * emb_dim, emb_dim))
        self.w_v = nn.Parameter(torch.FloatTensor(2 * emb_dim, emb_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        """
        Paramters
        ---------
        inputs: dict
            curr_node_id: torch.LongTensor [batch_size x 1]
            next_node_id: torch.LongTensor [batch_size x 1]
            node_feat: torch.FloatTensor [batch_size x num_nodes x node_dim]
            mask: torch.LongTensor [batch_size x num_nodes]
            state: torch.FloatTensor [batch_size x state_dim]

        Returns
        -------
        h: torch.tensor [batch_size x emb_dim]
            graph embeddings 
        """
        #----------------
        # input features
        #----------------
        curr_node_id = inputs["curr_node_id"]
        next_node_id = inputs["next_node_id"]
        node_feat = inputs["node_feats"]
        mask = inputs["mask"]
        state = inputs["state"]

        #---------------------------
        # initial linear projection
        #---------------------------
        node_emb  = self.init_linear_node(node_feat[:, 1:, :]) # [batch_size x num_loc x emb_dim]
        depot_emb = self.init_linear_depot(node_feat[:, 0:1, :2]) # [batch_size x 1 x emb_dim]
        new_node_feat = torch.cat((depot_emb, node_emb), 1) # [batch_size x num_nodes x emb_dim]
        new_node_feat = self.dropout(new_node_feat)

        #---------------
        # preprocessing
        #---------------
        batch_size = curr_node_id.size(0)
        curr_emb = new_node_feat.gather(1, curr_node_id[:, None, None].expand(batch_size, 1, self.emb_dim))
        next_emb = new_node_feat.gather(1, next_node_id[:, None, None].expand(batch_size, 1, self.emb_dim))
        if state is not None and self.state_dim > 0:
            state_emb = self.init_linear_state(state) # [batch_size x emb_dim]
            input_q  = torch.cat((curr_emb, next_emb, state_emb[:, None, :]), -1) # [batch_size x 1 x (3*emb_dim)]
        else:
            input_q  = torch.cat((curr_emb, next_emb), -1) # [batch_size x 1 x (2*emb_dim)]
        input_kv = torch.cat((curr_emb.expand_as(new_node_feat), new_node_feat), -1) # [batch_size x num_nodes x (2*emb_dim)]

        #--------------------
        # An attention layer
        #--------------------
        q = torch.matmul(input_q,  self.w_q) # [batch_size x 1 x emb_dim]
        k = torch.matmul(input_kv, self.w_k) # [batch_size x num_nodes x emb_dim]
        v = torch.matmul(input_kv, self.w_v) # [batch_size x num_nodes x emb_dim]
        compatibility = self.norm_factor * torch.matmul(q, k.transpose(-2, -1)) # [batch_size x 1 x num_nodes]
        compatibility[(~mask).unsqueeze(1).expand_as(compatibility)] = -math.inf
        attn = torch.softmax(compatibility, dim=-1)
        h = torch.matmul(attn, v) # [batch_size x 1 x emb_dim]
        h = h.squeeze(1) # [batch_size x emb_dim]

        return h