
# This is a sample Python script.
import random
import numpy as np
from connector.dataloader.graph.homogeneous import GraphLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#import matplotlib.pyplot as plt
import networkx as nx
from connector.models.struc2vec import Struc2Vec
from  arguments import parse_args



def parse_args1():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input', help = 'Input graph file', default = 'data/karate/karate.edgelist')
    parser.add_argument('--label-file', default='data/cora/cora_labels.txt')
    parser.add_argument('--feature-file',default='data/cora/cora.features')
    parser.add_argument('--graph-format',default='edgelist', choices=['adjlist', 'edgelist'])
    parser.add_argument()
    parser.add_argument('--weighted', action= 'store_true')
    args = parser.parse_args()

    return args


def train(args):
    args.graph_format = "edgelist"
    args.dataset = 'karate'
    G = GraphLoader(data_dir=args.data_dir, dataset=args.dataset)

    print("Reading")
    if args.graph_format == 'adjlist':
        G.read_adj_graph()
    elif args.graph_format == 'edgelist':
        G.read_edgelist()

    #nx.draw(g.G, with_labels=True)
    #plt.savefig("filename.png")

    model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )  # init model
    model.train(window_size=5, iter=3)  # train model
    embeddings = model.get_embeddings()  # get embedding vectors

    print(embeddings)



def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()




