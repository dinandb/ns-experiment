from abc import ABC, abstractmethod
import igraph as ig
from HMI import nhmi


class SuperAlgorithm(ABC):
    """Abstract base class for hierarchical clustering algorithms."""

    @abstractmethod
    def run(self, g: ig.Graph) -> list[tuple[int, int]]:
        """Run the algorithm and return a merge list (agglomerative sequence).

        Args:
            g: igraph.Graph object

        Returns:
            list of (int, int) tuples representing merges in agglomerative format.
            Merge k produces internal node n_nodes + k.
        """
        pass

    def optimize_for_modularity(g: ig.Graph, merges: list[tuple[int, int]]) -> tuple[list[list[int]], float]:
        """Cut the dendrogram at the level that maximises modularity on g.

        Returns the best partition (list of communities, each a sorted list of
        node IDs) and its modularity score.
        """
        n = g.vcount()
        clusters: dict[int, set[int]] = {i: {i} for i in range(n)}

        best_mod = -1.0
        best_partition: list[list[int]] = [list(range(n))]

        for step, (a, b) in enumerate(merges):
            clusters[n + step] = clusters.pop(a) | clusters.pop(b)

            membership = [0] * n
            for comm_id, nodes in enumerate(clusters.values()):
                for node in nodes:
                    membership[node] = comm_id

            mod = g.modularity(membership)
            print(f"modularity of")
            if mod > best_mod:
                best_mod = mod
                best_partition = [sorted(nodes) for nodes in clusters.values()]

        return best_partition, best_mod


# class CombinedAlgorithm(SuperAlgorithm):
#     """Runs multiple algorithms and selects the best based on nHMI consensus."""

#     def __init__(self, algorithms=None):
#         """Initialize with list of algorithm instances.

#         Args:
#             algorithms: list of SuperAlgorithm instances.
#                        Defaults to [GirvanNewman(), ClusteringCoefficientDivisive()]
#         """
#         if algorithms is None:
#             from algorithms.divisive.girvan_newman import GirvanNewman
#             from algorithms.divisive.clust_coef_divisive import ClusteringCoefficientDivisive
#             from algorithms.agglomerative.linkage import Linkage
#             from algorithms.agglomerative.cosine_similarity import CosineSimilarity
#             algorithms = [GirvanNewman(), ClusteringCoefficientDivisive(), Linkage(), CosineSimilarity()]
#             self.algorithms = algorithms

#     def run(self, g: ig.Graph) -> list[tuple[int, int]]:
#         """Run all algorithms and return the merge list with highest average nHMI to others."""
#         n = g.vcount()

#         # Run all algorithms
#         merge_lists = [algo.run(g) for algo in self.algorithms]

#         # If only one algorithm, return it
#         if len(merge_lists) == 1:
#             return merge_lists[0]

#         # Score each merge list by average nHMI similarity to all others
#         scores = []
#         for i, merges_i in enumerate(merge_lists):
#             total_similarity = 0.0
#             for j, merges_j in enumerate(merge_lists):
#                 if i != j:
#                     total_similarity += nhmi(n, merges_i, merges_j, verbose=False)
#             avg_similarity = total_similarity / (len(merge_lists) - 1)
#             scores.append(avg_similarity)

#         # Return the merge list with highest average similarity
#         best_idx = scores.index(max(scores))
#         return merge_lists[best_idx]


# class SingleAlgorithm(SuperAlgorithm):
#     """Runs a single algorithm."""

#     def __init__(self, algorithm):
#         """Initialize with an algorithm instance.

#         Args:
#             algorithm: SuperAlgorithm instance to run.
#         """
#         self.algorithm = algorithm

#     def run(self, g: ig.Graph) -> list[tuple[int, int]]:
#         """Run the algorithm and return the merge list."""
#         return self.algorithm.run(g)
