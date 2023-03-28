import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, feed_dim, dropout, activation=nn.GELU()):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(model_dim, feed_dim)
        self.linear2 = nn.Linear(feed_dim, model_dim)

        self.activation = activation

    def forward(self, x):

        return self.linear2(self.activation(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):

        return x + self.dropout(layer(self.norm(x)))
