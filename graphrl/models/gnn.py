import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from graphrl.models.base import BaseModel
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):
    def __init__(self, dim, bias=True):
        super(ConvolutionalLayer, self).__init__()
        self.dim = dim
        self.weight = Parameter(torch.FloatTensor(dim, dim))

        if bias:
            self.bias = Parameter(torch.FloatTensor(dim))
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

class AttentiveLayer(nn.Module):
    def __init__(self, dim, dropout=0, activation=nn.LeakyReLU()):
        super(AttentiveLayer, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.activation = activation

        self.weight = Parameter(torch.FloatTensor(dim, dim))
        self.a = Parameter(torch.FloatTensor(2*dim, 1))
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

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)

        aggregated_features = torch.spmm(attention, Wh)

        return aggregated_features

    def forward(self, input, adj_matrix):
        output = self.aggregate_with_attention(input, adj_matrix)

        return output

    

class GCN(BaseModel):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout=0, hidden_layer_activation=nn.LeakyReLU(), output_layer_activation=nn.LogSoftmax()):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.message_dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.activation_modules = nn.ModuleList()
        self.normalize = nn.LayerNorm(self.num_classes)
        
        self.conv_layers = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.conv_layers.append(ConvolutionalLayer(self.num_features))

        if self.num_layers < 2:
            self.linears.append(nn.Linear(self.num_features, self.num_classes))
        else:
            self.linears.append(nn.Linear(self.num_features, self.hidden_dim))
            self.activation_modules.append(hidden_layer_activation)

            for layer in range(self.num_layers - 2):
                self.conv_layers.append(ConvolutionalLayer(self.hidden_dim))
                self.linears.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.activation_modules.append(hidden_layer_activation)

            self.conv_layers.append(ConvolutionalLayer(self.hidden_dim))
            self.linears.append(nn.Linear(self.hidden_dim, self.num_classes))

        self.activation_modules.append(output_layer_activation)

    def forward(self, x, adj_matrix):
        for layer in range(self.num_layers):
            agregated_features = self.conv_layers[layer](x, adj_matrix)
            x = x + agregated_features
            x = self.linears[layer](x)
            x = self.activation_modules[layer](x)

        output = self.message_dropout(self.normalize(x))

        return output
        

class GraphSAGE(BaseModel):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout=0, hidden_layer_activation=nn.LeakyReLU(), output_layer_activation=nn.LogSoftmax()):
        super(GraphSAGE, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.message_dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.activation_modules = nn.ModuleList()
        self.normalize = nn.LayerNorm(self.num_classes)
        
        self.conv_layers = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.conv_layers.append(ConvolutionalLayer(self.num_features))

        if self.num_layers < 2:
            self.linears.append(nn.Linear(self.num_features * 2, self.num_classes))
        else:
            self.linears.append(nn.Linear(self.num_features * 2, self.hidden_dim))
            self.activation_modules.append(hidden_layer_activation)

            for layer in range(self.num_layers - 2):
                self.conv_layers.append(ConvolutionalLayer(self.hidden_dim))
                self.linears.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
                self.activation_modules.append(hidden_layer_activation)

            self.conv_layers.append(ConvolutionalLayer(self.hidden_dim))
            self.linears.append(nn.Linear(self.hidden_dim * 2, self.num_classes))
        
        self.activation_modules.append(output_layer_activation)

    def forward(self, x, adj_matrix):
        for layer in range(self.num_layers):
            agregated_features = self.conv_layers[layer](x, adj_matrix)
            x = torch.cat([x, agregated_features], dim=1)
            x = self.linears[layer](x)
            x = self.activation_modules[layer](x)

        output = self.message_dropout(self.normalize(x))

        return output

class GIN(BaseModel):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout=0, hidden_layer_activation=nn.LeakyReLU(), output_layer_activation=nn.LogSoftmax()):
        super(GIN, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.message_dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(self.num_classes)
        self.num_layers = num_layers

        self.activation_modules = nn.ModuleList()
        
        self.conv_layers = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.conv_layers.append(ConvolutionalLayer(self.num_features))
        self.linears.append(nn.Linear(self.num_features, self.hidden_dim))
        self.activation_modules.append(hidden_layer_activation)

        if self.num_layers > 1:
            for layer in range(self.num_layers - 1):
                self.conv_layers.append(ConvolutionalLayer(self.hidden_dim))
                self.linears.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.activation_modules.append(hidden_layer_activation)
        
        self.out_linear = torch.nn.Linear(self.num_features + hidden_dim*self.num_layers, self.num_classes)
        self.output_activation = output_layer_activation

    def forward(self, x, adj_matrix):
        hidden_states = [x]
        for layer in range(self.num_layers):
            agregated_features = self.conv_layers[layer](x, adj_matrix)
            x = self.linears[layer](agregated_features)
            x = self.activation_modules[layer](x)
            hidden_states.append(x)

        output = torch.cat(hidden_states, dim=1)
        output = self.output_activation(self.out_linear(output))
        output = self.message_dropout(self.normalize(output))

        return output

class GAT(BaseModel):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout=0, hidden_layer_activation=nn.LeakyReLU(), output_layer_activation=nn.LogSoftmax()):
        super(GAT, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.message_dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.activation_modules = nn.ModuleList()
        self.normalize = nn.LayerNorm(self.num_classes)
        
        self.conv_layers = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.conv_layers.append(AttentiveLayer(self.num_features))

        if self.num_layers < 2:
            self.linears.append(nn.Linear(self.num_features, self.num_classes))
        else:
            self.linears.append(nn.Linear(self.num_features, self.hidden_dim))
            self.activation_modules.append(hidden_layer_activation)

            for layer in range(self.num_layers - 2):
                self.conv_layers.append(AttentiveLayer(self.hidden_dim))
                self.linears.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.activation_modules.append(hidden_layer_activation)

            self.conv_layers.append(AttentiveLayer(self.hidden_dim))
            self.linears.append(nn.Linear(self.hidden_dim, self.num_classes))

        self.activation_modules.append(output_layer_activation)

    def forward(self, x, adj_matrix):
        for layer in range(self.num_layers):
            agregated_features = self.conv_layers[layer](x, adj_matrix)
            x = x + agregated_features
            x = self.linears[layer](x)
            x = self.activation_modules[layer](x)

        output = self.message_dropout(self.normalize(x))

        return output