# This is a sample Python script.
import random
import numpy as np
from connector.dataloader.graph.homogeneous import GraphLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#import matplotlib.pyplot as plt
import networkx as nx
from connector.models.node2vec import Node2vec
from  arguments import parse_args


def train(args):
    args.graph_format = "edgelist"
    args.dataset = 'cora'

    g = GraphLoader(data_dir=args.data_dir, dataset=args.dataset)

    print("Reading")
    if args.graph_format == 'adjlist':
        g.read_adj_graph()
    elif args.graph_format == 'edgelist':
        g.read_edgelist()

    #nx.draw(g.G, with_labels=True)
    #plt.savefig("filename.png")
    model = Node2vec(graph=g, path_length=5,
                     num_paths=10, dim=12,
                                  workers = 1, p=1, q=1, window=3)
    print(model.vectors)


def main():
    args = parse_args()
    train(args)

    
if __name__ == '__main__':
    main()
