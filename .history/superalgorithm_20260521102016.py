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


class SingleAlgorithm(SuperAlgorithm):
    """Runs a single algorithm."""

    def __init__(self, algorithm):
        """Initialize with an algorithm function.

        Args:
            algorithm: function that takes igraph.Graph and returns merges.
        """
        self.algorithm = algorithm

    def run(self, g: ig.Graph) -> list[tuple[int, int]]:
        """Run the algorithm and return the merge list."""
        return self.algorithm(g)