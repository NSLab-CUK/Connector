import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from graphrl.models.base import BaseModel
from graphrl.utils.model_utils import VECTOR_FUNCTIONS



class SiNE(BaseModel):
    def __init__(self, num_nodes, dim1, dim2):
        super(SiNE, self).__init__()
        self.activation = nn.Tanh()
        self.embeddings = nn.Embedding(num_nodes, dim1)
        self.layer_11 = nn.Linear(dim1, dim2, bias=False)
        self.layer_12 = nn.Linear(dim1, dim2, bias=False)

        self.bias_1 = Parameter(torch.zeros(1))
        self.layer_2 = nn.Linear(dim2, 1, bias=False)
        self.bias_2 = Parameter(torch.zeros(1))

        self.register_parameter('bias_1', self.bias_1)
        self.register_parameter('bias_2', self.bias_2)
        

    def forward(self, x_i, x_j, x_k, delta):
        '''
            Calculate the loss of the model by comparing the postive and negative edges
            Compute 2 layers feed forward
            x_i: the observed node
            x_j: make the positive edge with x_i
            x_k: make the negative edge with x_i
            delta: the hyper parameters to control the comparison of postive and negative value
        '''
        i_emb = self.embeddings(x_i)
        j_emb = self.embeddings(x_j)
        k_emb = self.embeddings(x_k)

        layer_11 = self.activation(self.layer_11(i_emb) + self.layer_12(j_emb) + self.bias_1)
        layer_12 = self.activation(self.layer_11(i_emb) + self.layer_12(k_emb) + self.bias_1)
        
        positive_edge = self.activation(self.layer_2(layer_11) + self.bias_2)
        negative_edge = self.activation(self.layer_2(layer_12) + self.bias_2)

        zeros = Variable(torch.zeros(1))

        loss = torch.max(zeros, positive_edge + delta - negative_edge)

        return torch.sum(loss)

    def _regularizer(self, x):
        zeros = torch.zeros_like(x)
        normalization = torch.norm(x - zeros, p=2)
        return torch.pow(normalization, 2)

    def regularize_weights(self):
        loss = 0

        for param in self.parameters():
            loss += self._regularizer(param)

        return loss

    def get_embedding(self, x):
        x = Variable(torch.LongTensor([x]))

        embedding = self.embeddings(x)
        
        return embedding.data.numpy()[0]

    def get_edge_features(self, x, y, operator='hadamard'):
        func = VECTOR_FUNCTIONS[operator]

        x_emb = self.get_embedding(x)
        y_emb = self.get_embedding(y)

        return func(x_emb, y_emb)