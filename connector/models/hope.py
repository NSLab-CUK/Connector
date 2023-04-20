import networkx as nx
import scipy.sparse.linalg as sla
import scipy.linalg as la
import numpy as np
from sklearn.preprocessing import normalize
import torch

from connector.models.base import BaseModel

class HOPE(BaseModel):
    def __init__(self, args):
        super(HOPE, self).__init__()
        self.graph = None
        self.X1 = None
        self.X2 = None
        self.X = None

        self.dim = args.dim
        self.args = args

    def calculate_score(self, measurement):
        n = self.graph.num_nodes

        A = self.graph.adjmatrix(directed=True, weighted=False)

        if measurement == 'katz': # Katz
            score = (la.inv(np.identity(n) - self.args.beta * A) - np.identity(n))
        elif measurement == 'cn':  # Common Neighbors
            score = np.matmul(A, A)
        elif measurement == 'rpr':  # Rooted PageRank
            P = self.graph.adjmat(directed=True, weighted=False, scaled=1)  # scaled=0 in paper but possibly wrong?
            score = (1 - self.args.alpha) * la.inv(np.eye(n) - self.args.alpha * P)
        else: # Adamic-Adar
            D = np.eye(n)
            for i in range(n):
                k = sum(A[i][j] + A[j][i] for j in range(n))
                D[i][i] /= k
            score = np.matmul(np.matmul(A, D), A)
        
        return score

    def forward(self, graph, measurement):
        self.graph = graph

        score = self.calculate_score(measurement)

        u, s, vt = sla.svds(score, k=self.dim // 2)

        sigma = np.sqrt(s)
        self.X1 = normalize(u * sigma)
        self.X2 = normalize(vt.T * sigma)

        self.X = torch.cat((torch.tensor(self.X1), torch.tensor(self.X2)), dim=1)

        return self.X



