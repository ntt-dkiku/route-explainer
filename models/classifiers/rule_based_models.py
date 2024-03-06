import torch
import torch.nn as nn

TOUR_LENGTH = 0
TIME_WINDOW = 1

class kNearestPredictor(nn.Module):
    def __init__(self, problem, k, k_type):
        """
        Paramters
        ---------
        problem: str
            problem type
        k: float
            if the vehicle visis k% nearest node, this model labels the visit as prioritizing tour length
        """
        super().__init__()
        self.problem = problem
        self.num_classes = 2
        self.k_type = k_type
        if k_type == "num":
            self.k = int(k)
        elif k_type == "ratio":
            self.k = k
        else:
            assert False, "Invalid k_type. select from [num, ratio]"

    def forward(self, inputs):
        """
        Parameters
        ----------

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

        coord_dim = 2
        batch_size = curr_node_id.size(0)
        coords = node_feat[:, :, :coord_dim] # [batch_size x num_nodes x coord_dim]
        num_candidates = (mask > 0).sum(dim=-1) # [batch_size]
        topk = torch.round(num_candidates * self.k).to(torch.long) # [batch_size]
        curr_coord = coords.gather(1, curr_node_id[:, None, None].expand_as(coords)) # [batch_size x 1 x coord_dim]
        dist_from_curr_node = torch.norm(curr_coord - coords, dim=-1) # [batch_size x 1 x num_nodes]
        visit_topk = []
        for i in range(batch_size):
            if self.k_type == "num":
                k = self.k
            else:    
                k = topk[i].item()
            id = torch.topk(input=dist_from_curr_node[i], k=k, dim=-1, largest=True)[1]
            visit_topk.append(torch.isin(next_node_id[i], id))
        visit_topk = torch.stack(visit_topk, 0)
        idx = (1 - visit_topk.int()).to(torch.long)
        probs = torch.zeros(batch_size, self.num_classes).to(torch.float)
        probs.scatter_(-1, idx.unsqueeze(-1).expand_as(probs), 1.0)
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
        if isinstance(node_feats, np.ndarray):
            node_feats = torch.from_numpy(node_feats.astype(np.float32)).clone()
        tour = torch.LongTensor(tour)
        coord_dim = 2
        out = {"curr_node_id": [], "next_node_id": [], "mask": [], "state": []}
        for step in range(first_explained_step, len(tour) - 1):
            # node ids
            curr_node_id = tour[step]
            next_node_id = tour[step + 1]
            # mask & state
            max_coord = 100
            coord = node_feats[:, coord_dim] / max_coord # [num_nodes x coord_dim]
            time_window = node_feats[:, coord_dim:] # [num_nodes x 2(start, end)]
            time_window = (time_window - time_window[1:].min()) / time_window[1:].max() # min-max normalization
            curr_time = torch.FloatTensor([0.0])
            raw_coord = node_feats[:, coord_dim]
            raw_time_window = node_feats[:, coord_dim:]
            raw_curr_time = torch.FloatTensor([0.0])
            mask = torch.ones(node_feats.size(0), dtype=torch.long) # feasible -> 1, infeasible -> 0
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
        node_feats = node_feats.unsqueeze(0).expand(out["mask"].size(0), node_feats.size(-2), node_feats.size(-1))
        out.update({"node_feats": node_feats})
        return out

    def get_topk_ids(self, input, k, dim, largest):
        """
        Parameters
        ----------
        input: torch.tensor [batch_size x num_nodes x num_nodes]
        k: torch.tensor [batch_size]
        dim: int
        largest: bool

        Returns
        -------
        topk_ids: torch.tensor [batch_size x num_node x k]
        """
        batch_size = input.size(0)
        max_k = k.max()
        ids = []
        for i in range(batch_size):
            id = torch.topk(input=input[i], k=k[i].item(), dim=dim, largest=largest)[1]

            # adjust tensor size
            if id.size(0) == 0:
                id = torch.full((max_k, ), -1000)
            elif id.size(0) < max_k:
                id = torch.cat((id, torch.full((max_k - id.size(0), ), id[0])), -1)
            ids.append(id)
        return torch.stack(ids, 0)