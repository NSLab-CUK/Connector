import numpy as np
import torch
from torch.autograd import Variable
import networkx as nx

class NodeEmbedding(object):
    def __init__(self, graph):
        self.graph = graph
        self._node2id = {}
        self._id2node = {}
        self.current_id = 0
        self.load_vocab()

    def load_vocab(self, graph=None):
        if graph is None:
            graph = self.graph
        for node in graph.nodes():
            if node not in self._node2id:
                self.current_id += 1
                self._node2id[node] = self.current_id
                self._id2node[self.current_id] = node  

    def node2id(self, node):
        return self._node2id[node]

    def id2node(self, id):
        return self._id2node[id]

    def ___len___(self):
        return self.current_id

class SignedGraph(object):
    def __init__(self, positive_graph=None, negative_graph=None):
        self.positive_graph = positive_graph
        self.negative_graph = negative_graph
        self.node_embedding = NodeEmbedding(positive_graph)
        self.node_embedding.load_vocab(negative_graph)
        
    def get_positive_edges(self):
        return self.positive_graph.edges()

    def get_negative_edges(self):
        return self.negative_graph.edges()

    def ___len___(self):
        return self.node_embedding.___len___()

    @property
    def num_nodes(self):
        return self.___len___()

    def get_triplets(self, p0=True, ids=True):
        triplets = []

        for n_i in self.positive_graph.nodes():
            for n_j in self.positive_graph[n_i]:
                if n_j in self.negative_graph:
                    for n_k in self.negative_graph[n_j]:
                        a, b, c =  n_i, n_j, n_k

                        if ids:
                            a = self.node_embedding.node2id(a)
                            b = self.node_embedding.node2id(b)
                            c = self.node_embedding.node2id(c)

                elif p0:
                    a, b = n_i, n_j
                    c = 0

                    if ids:
                        a = self.node_embedding.node2id(a)
                        b = self.node_embedding.node2id(b)

                triplets.append([a, b, c])

        triplets = np.array(triplets)
        return triplets

    @staticmethod
    def tensorfy_col(x, col_idx):
        col = x[:, col_idx]
        col = torch.LongTensor(col)
        return Variable(col)

    @staticmethod
    def sample_batch(triplets, batch_size):
        n_rows = triplets.shape[0]
        rows = np.random.choice(n_rows, batch_size, replace=False)
        batch = triplets[rows, :]

        n_i = SignedGraph.tensorfy_col(batch, 0)
        n_j = SignedGraph.tensorfy_col(batch, 1)
        n_k = SignedGraph.tensorfy_col(batch, 2)

        return n_i, n_j, n_k

    @staticmethod
    def load_from_file(file_path, delimiter='\t', directed=False):
        positive_graph = nx.DiGraph() if directed else nx.Graph()
        negative_graph = nx.DiGraph() if directed else nx.Graph()
        
        file = open(file_path)

        for line in file:
            u, v, w = line.strip().split(delimiter)
            w = float(w)

            if w > 0:
                positive_graph.add_edge(u, v, weight=w)
            elif w < 0:
                negative_graph.add_edge(u, v, weight=w)

        file.close()

        return SignedGraph(positive_graph, negative_graph)
        