import igraph as ig
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from superalgorithm import SuperAlgorithm

class CosineSimilarity(SuperAlgorithm):
    """Agglomerative community detection based on the cosine similarity of the adjacencent matrix rows"""

    def run(self, g: ig.Graph, linkage_method='average') -> list[tuple[int, int]]:
        adj = np.asarray(g.get_adjacency().data)
        dist = pdist(adj, metric='cosine')
        # NaN is converted to 1.0, the worst possible value.
        # Other options are also possible, maybe options that require less memory usage
        dist = np.nan_to_num(dist, nan=1.0)
        D = linkage(dist, method=linkage_method, metric='precomputed')
        merges = [(a, b) for a, b, _, _ in D]

        return merges