import math
import torch
import torch.nn as nn

class MLPNodeEncoder(nn.Module):
    def __init__(self, coord_dim, node_dim, emb_dim, num_mlp_layers, dropout):
        super().__init__()
        self.coord_dim = coord_dim
        self.node_dim  = node_dim 
        self.emb_dim   = emb_dim
        self.num_mlp_layers = num_mlp_layers

        # initial embedding
        self.init_linear_nodes = nn.Linear(node_dim, emb_dim)
        self.init_linear_depot = nn.Linear(coord_dim, emb_dim)

        # MLP Encoder
        self.mlp = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.mlp.append(nn.Linear(emb_dim, emb_dim, bias=True))

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        """
        Paramters
        ---------
        inputs: dict
            node_feat: torch.FloatTensor [batch_size x num_nodes x node_dim]

        Returns
        -------
        node_emb: torch.tensor [batch_size x num_nodes x emb_dim]
            node embeddings 
        """
        #----------------
        # input features
        #----------------
        node_feat = inputs["node_feats"]

        #------------------------------------------------------------------------
        # initial linear projection for adjusting dimensions of locs & the depot
        #------------------------------------------------------------------------
        # node_feat = self.dropout(node_feat)
        loc_emb  = self.init_linear_nodes(node_feat[:, 1:, :]) # [batch_size x num_loc x emb_dim]
        depot_emb = self.init_linear_depot(node_feat[:, 0:1, :2]) # [batch_size x 1 x emb_dim]
        node_emb = torch.cat((depot_emb, loc_emb), 1) # [batch_size x num_nodes x emb_dim]

        #--------------
        # MLP encoding
        #--------------
        for i in range(self.num_mlp_layers):
            # node_emb = self.dropout(node_emb)
            node_emb = self.mlp[i](node_emb)
            if i != self.num_mlp_layers - 1:
                node_emb = torch.relu(node_emb)
        return node_emb