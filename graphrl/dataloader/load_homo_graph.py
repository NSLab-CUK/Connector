from abc import ABC

import networkx as nx
from torch.utils.data import Dataset
import numpy as np

class GraphLoader(Dataset, ABC):
    def __init__(self, data_dir, dataset, directed=False, weighted=False, is_sparse=False, device=None):
        super(GraphLoader, self).__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.graph = None
        self.directed = directed
        self.weighted = weighted
        self.is_sparse = is_sparse
        self.device = device

        self.edgelist_file = f"{self.data_dir}/{self.dataset}/edgelist.txt"
        self.labels_file = f"{self.data_dir}/{self.dataset}/labels.txt"
        self.features_file = f"{self.data_dir}/{self.dataset}/features.txt"
        self.adj_file = f"{self.data_dir}/{self.dataset}/adjlist.txt"

    def read_adj_graph(self, filename=None):
        if filename is not None:
            self.adj_file = filename

        self.graph = nx.read_adjlist(self.adj_file, create_using=nx.DiGraph(), nodetype=int, data=True)

        for i, j in self.graph.edges():
            self.graph.edges[i, j]['weight'] = 1

    def read_edgelist(self, filename=None):
        if filename is not None:
            self.edgelist_file = filename

        self.graph = nx.DiGraph()
        file = open(self.edgelist_file, 'r')

        line = file.readline()
        while line:
            if self.weighted:
                src, dst, w = line.split()
            else:
                src, dst = line.split()
                w = 1

            self.graph.add_edge(src, dst)
            self.graph[src][dst]['weight'] = float(w)

            if not self.directed:
                self.graph.add_edge(src, dst)
                self.graph[dst][src]['weight'] = float(w)

            line = file.readline()

        file.close() 

    def adjmatrix(self, directed=False, weighted=False, scaled=None, sparse=False):
        G = self.graph
        if not self.directed and not directed:
            G = nx.to_undirected(G)
        A = nx.adjacency_matrix(G).astype(np.float32)
        if not sparse:
            A = np.array(nx.adjacency_matrix(G).todense())
        if not self.weighted and not weighted:
            A = A.astype(np.bool8).astype(np.float32)
        if scaled is not None:  # e.g. scaled = 1
            A = A / A.sum(scaled, keepdims=True)
        return A

    @property
    def num_nodes(self):
        return self.graph.number_of_nodes()
    


