import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Node encoder
from models.classifiers.nn_classifiers.encoders.mlp_node_encoder import MLPNodeEncoder
from models.classifiers.nn_classifiers.encoders.mha_node_encoder import SelfMHANodeEncoder
# Edge encoder
from models.classifiers.nn_classifiers.encoders.attn_edge_encoder import AttentionEdgeEncoder
from models.classifiers.nn_classifiers.encoders.concat_edge_encoder import ConcatEdgeEncoder
# Decoder
from models.classifiers.nn_classifiers.decoders.lstm_decoder import LSTMDecoder
from models.classifiers.nn_classifiers.decoders.mlp_decoder import MLPDecoder
from models.classifiers.nn_classifiers.decoders.mha_decoder import SelfMHADecoder

# data loader
from utils.data_utils.tsptw_dataset import load_tsptw_sequentially
from utils.data_utils.pctsp_dataset import load_pctsp_sequentially
from utils.data_utils.pctsptw_dataset import load_pctsptw_sequentially
from utils.data_utils.cvrp_dataset import load_cvrp_sequentially

NODE_ENC_LIST = ["mlp", "mha"]
EDGE_ENC_LIST = ["concat", "attn"]
DEC_LIST = ["mlp", "lstm", "mha"]

class NNClassifier(nn.Module):
    def __init__(self, 
                 problem: str, 
                 node_enc_type: str, 
                 edge_enc_type: str, 
                 dec_type: str, 
                 emb_dim: int, 
                 num_enc_mlp_layers: int, 
                 num_dec_mlp_layers: int, 
                 num_classes: int, 
                 dropout: float,
                 pos_encoder: str = "sincos"):
        super().__init__()
        self.problem = problem
        self.node_enc_type = node_enc_type
        self.edge_enc_type = edge_enc_type
        self.dec_type = dec_type
        assert node_enc_type in NODE_ENC_LIST, f"Invalid enc_type. select from {NODE_ENC_LIST}"
        assert dec_type in DEC_LIST, f"Invalid dec_type. select from {DEC_LIST}"
        self.is_sequential = True if dec_type in ["lstm", "mha"] else False
        coord_dim = 2 # only support 2d problem
        if problem == "tsptw":
            node_dim  = 4 # coords (2) + time window (2)
            state_dim = 1 # current time (1)
        elif problem == "pctsp":
            node_dim  = 4 # coords (2) + prize (1) + penalty (1)
            state_dim = 2 # current prize (1) + current penalty (1)
        elif problem == "pctsptw":
            node_dim  = 6 # coords (2) + prize (1) + penalty (1) + time window (2)
            state_dim = 3 # current prize (1) + current penalty (1) + current time (1)
        elif problem == "cvrp":
            node_dim  = 3 # coords (2) + demand (1)
            state_dim = 1 # remaining capacity (1)
        else:
            NotImplementedError

        #----------------
        # Graph encoding
        #----------------
        # Node encoder
        if node_enc_type == "mlp":
            self.node_enc = MLPNodeEncoder(coord_dim, node_dim, emb_dim, num_enc_mlp_layers, dropout)
        elif node_enc_type == "mha":
            num_heads = 8
            num_mha_layers = 2
            self.node_enc = SelfMHANodeEncoder(coord_dim, node_dim, emb_dim, num_heads, num_mha_layers, dropout)
        else:
            raise NotImplementedError
        
        # Readout
        if edge_enc_type == "concat":
            self.readout = ConcatEdgeEncoder(state_dim, emb_dim, dropout)
        elif edge_enc_type == "attn":
            self.readout = AttentionEdgeEncoder(state_dim, emb_dim, dropout)
        else:
            raise NotImplementedError

        #------------------------
        # Classification Decoder
        #------------------------
        if dec_type == "mlp":
            self.decoder = MLPDecoder(emb_dim, num_dec_mlp_layers, num_classes, dropout)
        elif dec_type == "lstm":
            self.decoder = LSTMDecoder(emb_dim, num_dec_mlp_layers, num_classes, dropout)
        elif dec_type == "mha":
            num_heads = 8
            num_mha_layers = 2
            self.decoder = SelfMHADecoder(emb_dim, num_heads, num_mha_layers, num_classes, dropout, pos_encoder)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        """
        Paramters
        ---------
        inputs: dict
            curr_node_id: torch.LongTensor [batch_size x max_seq_length] if self.sequential else [batch_size]
            next_node_id: torch.LongTensor [batch_size x max_seq_length] if self.sequential else [batch_size]
            node_feat: torch.FloatTensor [batch_size x max_seq_length x num_nodes x node_dim] if self.sequential else [batch_size x num_nodes x node_dim]
            mask: torch.LongTensor [batch_size x max_seq_length x num_nodes] if self.sequential else [batch_size x num_nodes]
            state: torch.FloatTensor [batch_size x max_seq_length x state_dim] if self.sequential else [batch_size x state_dim]

        Returns
        -------
        probs: torch.tensor [batch_size x seq_length x num_classes] if self.sequential else [batch_size x num_classes]
            probabilities of classes
        """
        #-----------------
        # Encoding graphs 
        #-----------------
        if self.is_sequential:
            shp = inputs["curr_node_id"].size()
            inputs = {key: value.flatten(0, 1) for key, value in inputs.items()}
        node_emb = self.node_enc(inputs) # [(batch_size*max_seq_length) x emb_dim] if self.sequential else [batch_size x emb_dim]
        graph_emb = self.readout(inputs, node_emb)
        if self.is_sequential:
            graph_emb = graph_emb.view(*shp, -1) # [batch_size x max_seq_length x emb_dim]

        #----------
        # Decoding
        #----------
        probs = self.decoder(graph_emb)

        return probs
    
    def get_inputs(self, routes, first_explained_step, node_feats):
        node_feats_ = node_feats.copy()
        node_feats_["tour"] = routes
        if self.problem == "tsptw":
            seq_data = load_tsptw_sequentially(node_feats_)
        elif self.problem == "pctsp":
            seq_data = load_pctsp_sequentially(node_feats_)
        elif self.problem == "pctsptw":
            seq_data = load_pctsptw_sequentially(node_feats_)
        elif self.problem == "cvrp":
            seq_data = load_cvrp_sequentially(node_feats_)
        else:
            NotImplementedError
        
        def pad_seq_length(batch):
            data = {}
            for key in batch[0].keys():
                padding_value = True if key == "mask" else 0.0
                # post-padding
                data[key] = torch.nn.utils.rnn.pad_sequence([d[key] for d in batch], batch_first=True, padding_value=padding_value)
            pad_mask = torch.nn.utils.rnn.pad_sequence([torch.full((d["mask"].size(0), ), True) for d in batch], batch_first=True, padding_value=False)
            data.update({"pad_mask": pad_mask})
            return data
        instance = pad_seq_length(seq_data)
        return instance