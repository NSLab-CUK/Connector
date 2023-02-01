import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from graphrl.models.base import BaseModel


class ConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(ConvolutionalLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj_matrix):
        output = torch.spmm(adj_matrix, torch.mm(input, self.weight))

        if self.bias is not None:
            return output + self.bias
        
        return output

class GCN(BaseModel):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout, device="cpu"):
        super(GCN, self).__init__(device)
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers

        self.activation_modules = nn.ModuleList()
        self.norm_modules = nn.ModuleList()
        
        self.conv_layers = nn.ModuleList()
        if self.num_layers < 2:
            self.conv_layers.append(ConvolutionalLayer(hidden_dim, self.num_classes))
            self.norm_modules.append(nn.LayerNorm(self.num_classes))
            self.norm_modules.append(nn.LogSoftmax())
        else:
            self.conv_layers.append(ConvolutionalLayer(num_features, self.hidden_dim))
            self.norm_modules.append(nn.LayerNorm(hidden_dim))
            self.activation_modules.append(nn.LeakyReLU())
            for layer in range(self.num_layers - 2):
                self.conv_layers.append(ConvolutionalLayer(self.hidden_dim, self.hidden_dim))
                self.norm_modules.append(nn.LayerNorm(hidden_dim))
                self.activation_modules.append(nn.LeakyReLU())
            self.norm_modules.append(nn.LayerNorm(self.num_classes))
            self.activation_modules.append(nn.LogSoftmax())
            self.conv_layers.append(ConvolutionalLayer(self.hidden_dim, self.num_classes))

    def forward(self, x, adj_matrix):
        for layer in range(self.num_layers):
            x = self.conv_layers[layer](x, adj_matrix)
            x = self.activation_modules[layer](x)
            x = self.norm_modules[layer](x)

        return x
        

