import igraph as ig
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.manifold import SpectralEmbedding
from superalgorithm import SuperAlgorithm

class Spectral(SuperAlgorithm):
    """Agglomerative community detection with spectral embedding."""

    def run(self, g: ig.Graph, linkage_method='single') -> list[tuple[int, int]]:
        adj = np.asarray(g.get_adjacency().data)
        embedding = SpectralEmbedding(n_components=16, affinity='precomputed')
        S = embedding.fit_transform(adj)
        D = linkage(S, linkage_method)

        merges = [(a, b) for a, b, _, _ in D]

        return merges