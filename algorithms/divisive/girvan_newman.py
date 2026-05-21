import igraph as ig
import networkx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from superalgorithm import SuperAlgorithm


class GirvanNewman(SuperAlgorithm):
    """Girvan-Newman community detection algorithm."""

    def run(self, g: ig.Graph) -> list[tuple[int, int]]:
        """Run Girvan-Newman and return agglomerative merges compatible with HMI."""
        return _run_girvan_newman(g)


def _run_girvan_newman(g: ig.Graph) -> list[tuple[int, int]]:
    """
    Run Girvan-Newman on g and return agglomerative merges compatible with HMI.

    Returns a list of (a, b) pairs where leaf node IDs are 0..n-1 and
    internal node k is produced by merge index k - n.
    """
    n = g.vcount()
    clusters = g.community_edge_betweenness()
    merges_list = [(int(a), int(b)) for a, b in clusters.merges]

    # igraph's merges may not be in topological order, so sort them
    available = set(range(n))
    remaining = list(range(len(merges_list)))
    new_order = []

    while remaining:
        for i, old_step in enumerate(remaining):
            a, b = merges_list[old_step]
            if a in available and b in available:
                new_order.append(old_step)
                available.discard(a)
                available.discard(b)
                available.add(n + old_step)
                remaining.pop(i)
                break

    old_to_new_step = {old: new for new, old in enumerate(new_order)}

    def remap(idx):
        return idx if idx < n else n + old_to_new_step[idx - n]

    merges = []
    for new_step, old_step in enumerate(new_order):
        a = remap(merges_list[old_step][0])
        b = remap(merges_list[old_step][1])
        merges.append((a, b))

    return merges


def run(g: ig.Graph) -> list[tuple[int, int]]:
    """Run Girvan-Newman algorithm. Wrapper for backward compatibility."""
    algo = GirvanNewman()
    return algo.run(g)


if __name__ == "__main__":
    # g_networkx = networkx.gnm_random_graph(500, 800)
    # g = ig.Graph.Erdos_Renyi(n=500, m=800).clusters().giant()
    g = ig.Graph.Famous("Zachary")
    # g = ig.Graph(n=g_networkx.number_of_nodes(), edges=list(g_networkx.edges()))

    clusters = g.community_edge_betweenness()
    communities = clusters.as_clustering()
    print(communities)

    n = g.vcount()
    merges_list = [(int(a), int(b)) for a, b in clusters.merges]
    print(merges_list)

    available = set(range(n))
    remaining = list(range(len(merges_list)))
    new_order = []

    while remaining:
        for i, old_step in enumerate(remaining):
            a, b = merges_list[old_step]
            if a in available and b in available:
                new_order.append(old_step)
                available.discard(a)
                available.discard(b)
                available.add(n + old_step)
                remaining.pop(i)
                break

    old_to_new_step = {old: new for new, old in enumerate(new_order)}

    def remap(idx):
        return idx if idx < n else n + old_to_new_step[idx - n]

    sizes = {i: 1 for i in range(n)}
    linkage_matrix = []

    for new_step, old_step in enumerate(new_order):
        a = remap(merges_list[old_step][0])
        b = remap(merges_list[old_step][1])
        size = sizes[a] + sizes[b]
        linkage_matrix.append([a, b, new_step + 1, size])
        sizes[n + new_step] = size

    linkage_matrix = np.array(linkage_matrix, dtype=float)

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    n_nodes = g.vcount()
    dendrogram(linkage_matrix, ax=ax3, labels=list(range(n_nodes)))
    plt.title("Girvan-Newman dendrogram")
    plt.xlabel("nodes")
    plt.ylabel("merge step")
    plt.tight_layout()
    plt.show()