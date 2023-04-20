import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):
    def __init__(self, dim, bias=True):
        super(ConvolutionalLayer, self).__init__()
        self.linear = nn.Linear(dim, dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))

        nn.init.uniform_(self.linear.weight.data, -stdv, stdv)

    def forward(self, X, adj_matrix):
        output = torch.spmm(adj_matrix, self.linear(X))

        return output


class AttentiveLayer(nn.Module):
    def __init__(self, dim, dropout=0, activation=nn.LeakyReLU()):
        super(AttentiveLayer, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.activation = activation

        self.weight = Parameter(torch.FloatTensor(dim, dim))
        self.a = Parameter(torch.FloatTensor(2 * dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def aggregate_with_attention(self, Wh, adj_matrix):
        Wh1 = torch.spmm(Wh, self.a[:self.dim, :])
        Wh2 = torch.spmm(Wh, self.a[self.dim:, :])

        e = Wh1 + Wh2.T
        e = self.activation(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)

        aggregated_features = torch.spmm(attention, Wh)

        return aggregated_features

    def forward(self, input, adj_matrix):
        output = self.aggregate_with_attention(input, adj_matrix)

        return output
