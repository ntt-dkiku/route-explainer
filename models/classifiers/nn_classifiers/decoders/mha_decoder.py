import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # batch_first
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size x max_seq_length x embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SelfMHADecoder(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, num_mha_layers: int, num_classes: int, dropout: float, pos_encoder: str = None, max_len: int = 100):
        super().__init__()
        self.num_mha_layers = num_mha_layers
    
        # positional encoding
        self.pos_encoder_type = pos_encoder
        if pos_encoder == "sincos":
            self.pos_encoder = PositionalEncoding(d_model=emb_dim, dropout=dropout, max_len=max_len)

        # MHA blocks
        mha_layer = nn.TransformerEncoderLayer(d_model=emb_dim, 
                                               nhead=num_heads,
                                               dim_feedforward=emb_dim,
                                               dropout=dropout,
                                               batch_first=True)
        self.mha = nn.TransformerEncoder(mha_layer, num_layers=num_mha_layers)

        # linear projection for adjusting out_dim to num_classes
        self.out_linear = nn.Linear(emb_dim, num_classes, bias=True)

        # Initializing weights
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, edge_emb):
        """
        Paramters
        ---------
        graph_emb: torch.tensor [batch_size x max_seq_length x emb_dim]

        Returns
        -------
        probs: torch.tensor [batch_size x max_seq_length x num_classes]
            probabilities of classes
        """
        #---------------
        # MHA decoding
        #---------------
        if self.pos_encoder_type == "sincos":
            edge_emb = self.pos_encoder(edge_emb)
        h = self.mha(edge_emb, is_causal=True) # [batch_size x max_seq_length x emb_dim]
        logits = self.out_linear(h)
        probs = F.log_softmax(logits, dim=-1)
        return probs