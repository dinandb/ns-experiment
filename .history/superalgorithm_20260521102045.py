from abc import ABC, abstractmethod
import igraph as ig
from HMI import nhmi
from algorithms.girvan_newman import run as gn_run
from algorithms.clust_coef_divisive import run as cc_run


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

e merge list."""
        return self.algorithm(g)