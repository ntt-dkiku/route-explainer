import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#------------
# base class
#------------
class DecisionPredictorBase(nn.Module):
    def __init__(self, coord_dim, node_dim, state_dim, emb_dim, num_mlp_layers, num_classes, dropout):
        super().__init__()
        self.coord_dim = coord_dim
        self.node_dim  = node_dim 
        self.emb_dim   = emb_dim
        self.state_dim = state_dim
        self.num_mlp_layers = num_mlp_layers
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
        
        # MLP
        self.mlp = nn.ModuleList()
        for i in range(self.num_mlp_layers):
            self.mlp.append(nn.Linear(emb_dim, emb_dim, bias=True))
        self.mlp.append(nn.Linear(emb_dim, num_classes, bias=True))

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
            curr_node_id: torch.LongTensor [batch_size]
            next_node_id: torch.LongTensor [batch_size]
            node_feat: torch.FloatTensor [batch_size x num_nodes x node_dim]
            mask: torch.LongTensor [batch_size x num_nodes]
            state: torch.FloatTensor [batch_size x state_dim]

        Returns
        -------
        probs: torch.tensor [batch_size x num_classes]
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
        curr_emb = new_node_feat.gather(1, curr_node_id.unsqueeze(-1).expand(batch_size, 1, self.emb_dim))
        next_emb = new_node_feat.gather(1, next_node_id.unsqueeze(-1).expand(batch_size, 1, self.emb_dim))
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
        
        #---------------
        # MLP (decoder)
        #---------------
        for i in range(self.num_mlp_layers):
            h = self.dropout(h)
            h = torch.relu(self.mlp[i](h))
        h = self.dropout(h)
        logits = self.mlp[-1](h)
        probs = F.log_softmax(logits, dim=-1)
        return probs

    def get_inputs(self, tour, first_explained_step, node_feats):
        """
        For TSPTW
        TODO: refactoring

        Parameters
        ----------
        tour: list [seq_length]
        first_explained_step: int
        node_feats np.array [num_nodes x node_dim]

        Returns
        -------
        out: dict (key: data type [data_size])
            curr_node_id: torch.tensor [num_explained_paths]
            next_node_id: torch.tensor [num_explained_paths]
            node_feats: torch.tensor [num_explained_paths x num_nodes x node_dim]
            mask: torch.tensor [num_explained_paths x num_nodes]
            state: torch.tensor [num_explained_paths x state_dim]
        """
        node_feats = {
            key: torch.from_numpy(node_feat.astype(np.float32).copy()).clone()
                    if isinstance(node_feat, np.ndarray) else
                    torch.tensor([node_feat])
            for key, node_feat in node_feats.items()
        }
        if isinstance(tour, np.ndarray):
            tour = torch.from_numpy(tour.astype(np.long).copy()).clone()
        else:
            tour = torch.LongTensor(tour)
        
        out = {"curr_node_id": [], "next_node_id": [], "mask": [], "state": []}
        for step in range(first_explained_step, len(tour) - 1):
            # node ids
            curr_node_id = tour[step]
            next_node_id = tour[step + 1]
            # mask & state
            max_coord = node_feats["grid_size"]
            coord = node_feats["coords"] / max_coord # [num_nodes x coord_dim]
            time_window = node_feats["time_window"] # [num_nodes x 2(start, end)]
            time_window = (time_window - time_window[1:].min()) / time_window[1:].max() # min-max normalization
            curr_time = torch.FloatTensor([0.0])
            raw_coord = node_feats["coords"]
            raw_time_window = node_feats["time_window"]
            raw_curr_time = torch.FloatTensor([0.0])
            num_nodes = len(node_feats["coords"])
            mask = torch.ones(num_nodes, dtype=torch.long) # feasible -> 1, infeasible -> 0
            for i in range(step + 1):
                curr_id = tour[i]
                if i > 0:
                    prev_id = tour[i - 1]
                    raw_curr_time += torch.norm(raw_coord[curr_id] - raw_coord[prev_id])
                    curr_time += torch.norm(coord[curr_id] - coord[prev_id])
                    # visited?
                    mask[curr_id] = 0
                    # curr_time exceeds the time window?
                    mask[curr_time > time_window[:, 1]] = 0
            curr_time = (raw_curr_time - raw_time_window[1:].min()) / raw_time_window[1:].max() # min-max normalization
            out["curr_node_id"].append(curr_node_id)
            out["next_node_id"].append(next_node_id)
            out["mask"].append(mask)
            out["state"].append(curr_time)
        out = {key: torch.stack(value, 0) for key, value in out.items()}
        node_feats = {
            key: node_feat.unsqueeze(0).expand(out["mask"].size(0), *node_feat.size())
            for key, node_feat in node_feats.items()
        }
        out.update({"node_feats": node_feats})
        return out

#---------------
# general class
#---------------
class DecisionPredictor(DecisionPredictorBase):
    def __init__(self, problem, emb_dim, num_mlp_layers, num_classes, drop):
        coord_dim = 2
        self.problem = problem
        if problem == "tsptw":
            node_dim = coord_dim + 2 # + time_window(start, end)
            state_dim = 1 # current_time
        elif problem == "cvrp":
            node_dim = coord_dim + 1 # + demand
            state_dim = 1 # used_capacity
        elif problem == "cvrptw":
            node_dim = coord_dim + 1 + 2 # + demand + time_window(start, end)
            state_dim = 2 # used_capacity + current_time
        else:
            assert False, f"problem {problem} is not supported!"
        super().__init__(coord_dim, node_dim, state_dim, emb_dim, num_mlp_layers, num_classes, drop)