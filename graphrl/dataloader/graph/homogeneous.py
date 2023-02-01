from abc import ABC
import torch
import networkx as nx
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp
from graphrl.utils.graph_utils import encode_onehot
from graphrl.utils.matrix_utils import normalize_matrix, convert_sparse_matrix_to_sparse_tensor

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
                self.graph.add_edge(dst, src)
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

    def load_features(self, filename=None):
        if filename is not None:
            filename = f"{self.data_dir}/{self.dataset}/{filename}"

        self.index_features = np.genfromtxt(filename, dtype=np.dtype(str))
        self.features = normalize_matrix(sp.csr_matrix(self.index_features[:, 1:-1], dtype=np.float32))
        self.labels = encode_onehot(self.index_features[:, -1])

        self.features = torch.FloatTensor(np.array(self.features.todense()))
        self.labels = torch.LongTensor(np.where(self.labels)[1])

    def load_adj_matrix_from_edges(self, filename=None):
        if self.index_features is None:
            self.load_features()

        if filename is not None:
            filename = f"{self.data_dir}/{self.dataset}/{filename}"

        index_array = np.array(self.index_features[:, 0], dtype=np.int32)
        index_dict = {j:i for i, j in enumerate(index_array)}

        edges = np.genfromtxt(filename, dtype=np.int32)
        edges = np.array(list(map(index_dict.get, edges.flatten())), dtype=np.int32).reshape(edges.shape) 

        adj_matrix = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(self.labels.shape[0], self.labels.shape[0]), dtype=np.float32)

        adj_matrix = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)

        adj_matrix = normalize_matrix(adj_matrix)
        self.adj_matrix = convert_sparse_matrix_to_sparse_tensor(adj_matrix)


    @property
    def num_nodes(self):
        return self.graph.number_of_nodes()
    


