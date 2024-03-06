import numpy as np
from models.solvers.ortools.ortools_base import ORToolsBase

class ORToolsTSP(ORToolsBase):
    def __init__(self, large_value=1e+6, scaling=False):
        super().__init__(large_value, scaling)
    
    def scaling_feats(self, node_feats):
        return {
            key: (node_feat * self.large_value).astype(np.int64) 
                 if key == "coords" else 
                 node_feat
            for key, node_feat in node_feats.items()
        }