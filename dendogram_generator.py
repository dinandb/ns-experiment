# this program generates a Random Hierarchical Graph (HRG) using the pyhrg library, where
# the dendrogram is 'valid', i.e. each internal node has a probability that is at least its parent's probability

from pyhrg.hrg import Dendrogram
from random import shuffle
import networkx as nx

# Given number of vertices n, make a completely random hierarchical random graph
def make_rnd_dendrogram(n: int) -> Dendrogram:
    # initialize a dummy graph with n vertices and no edges.
    dummy_graph = nx.Graph()
    dummy_graph.add_nodes_from(range(0, n))

    # make a completely random hierarchical random graph with n graph nodes
    dendrogram: Dendrogram = Dendrogram(dummy_graph)
    dendrogram.initialize()

    return dendrogram