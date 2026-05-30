import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from superalgorithm import SuperAlgorithm


class Linkage(SuperAlgorithm):
    """Agglomerative Linkage community detection algorithm."""

    def __init__(self, linkage_algorithm='single'):
        self.linkage_algorithm = linkage_algorithm

    def run(self, g: ig.Graph) -> list[tuple[int, int]]:
        """Run Agglomerative Linkage and return agglomerative merges compatible with HMI."""
        return _run_agglomerative(g, self.linkage_algorithm)


def _run_agglomerative(g: ig.Graph, linkage_algorithm: str) -> list[tuple[int, int]]:
    """
    Run Agglomerative Linkage on g and return agglomerative merges compatible with HMI.

    Returns a list of (a, b) pairs where leaf node IDs are 0..n-1 and
    internal node k is produced by merge index k - n.
    """

    distances = np.array(g.distances())
    finite_max = np.max(distances[np.isfinite(distances)])
    distances[distances == np.inf] = finite_max * 100
    D = linkage(distances, method=linkage_algorithm)
    merges = [(a, b) for a, b, _, _ in D]

    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(D)
    # plt.show()
    return merges


def run(g: ig.Graph, a: str = 'single') -> list[tuple[int, int]]:
    """Run Agglomerative Linkage algorithm. Wrapper for backward compatibility."""
    algo = Linkage(a)
    return algo.run(g)