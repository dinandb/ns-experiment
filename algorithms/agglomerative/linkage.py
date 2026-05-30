import igraph as ig
import numpy as np
from scipy.cluster.hierarchy import linkage

from superalgorithm import SuperAlgorithm


class Linkage(SuperAlgorithm):
    """Agglomerative Linkage community detection algorithm."""

    def run(self, g: ig.Graph, linkage_algorithm='single') -> list[tuple[int, int]]:
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

        return merges