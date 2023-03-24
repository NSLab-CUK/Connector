import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation=nn.GELU()):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.activation = activation

    def forward(self, x):

        return self.linear2(self.activation(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):

        return x + self.dropout(layer(self.norm(x)))
