import torch
import torch.nn as nn
from ..utils.layer_utils import *
from attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, feed_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feedforward = PositionWiseFeedForward(d_model, feed_dim, dropout)
        self.input = AddNorm(d_model, dropout)
        self.output = AddNorm(d_model, dropout)

    def forward(self, x, mask):
        x = self.input(x, lambda _x: self.attention(_x, _x, _x), mask=mask)
        x = self.output(x, self.feedforward)

        return self.dropout(x)
