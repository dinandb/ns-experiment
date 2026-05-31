import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import igraph as ig
import networkx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from superalgorithm import SuperAlgorithm


class FastGirvanNewman(SuperAlgorithm):
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

    # clusters = g.community_edge_betweenness()

    # make clusters here
    merges_list = cluster_alg(g)

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

def _edge_betweenness(g: ig.Graph) -> list[float]:
    # the edge betweenness of an edge e is defined as the number of adjacent edges to e
    degrees = g.degree()
    return [degrees[e.source] + degrees[e.target] - 2 for e in g.es]


def cluster_alg(g: ig.Graph) -> list[tuple[int, int]]:
    """Fast Girvan-Newman: iteratively remove highest-betweenness edges
    while modularity keeps improving. Returns merges list for dendrogram."""
    g = g.copy()
    n = g.vcount()

    counter = g.ecount()
    modularity_before = 0.0
    modularity_after = g.modularity(g.components().membership)

    splits = []  # (frozenset_part1, frozenset_part2) for each split caused by an edge removal

    while counter > 0: # and modularity_after > modularity_before:
        eb = _edge_betweenness(g)
        

        best = eb.index(max(eb))
        u, v = g.es[best].source, g.es[best].target
        g.delete_edges(best)

        membership = g.components().membership
        if membership[u] != membership[v]:  # edge removal caused a split
            part1 = frozenset(i for i, m in enumerate(membership) if m == membership[u])
            part2 = frozenset(i for i, m in enumerate(membership) if m == membership[v])
            splits.append((part1, part2))

        counter -= 1

    # Build agglomerative merges by replaying splits in reverse
    comp_to_id = {frozenset({i}): i for i in range(n)}
    merges = []
    internal_id = n

    for part1, part2 in reversed(splits):
        merges.append((comp_to_id[part1], comp_to_id[part2]))
        comp_to_id[part1 | part2] = internal_id
        internal_id += 1

    return merges




def run(g: ig.Graph) -> list[tuple[int, int]]:
    """Run Girvan-Newman algorithm. Wrapper for backward compatibility."""
    algo = FastGirvanNewman()
    return algo.run(g)


if __name__ == "__main__":
    # g_networkx = networkx.gnm_random_graph(500, 800)
    # g = ig.Graph.Erdos_Renyi(n=500, m=800).clusters().giant()
    g = ig.Graph.Famous("Zachary")
    # g = ig.Graph(n=g_networkx.number_of_nodes(), edges=list(g_networkx.edges()))

    # clusters = g.community_edge_betweenness()
    # communities = clusters.as_clustering()
    # print(communities)

    n = g.vcount()
    merges_list = cluster_alg(g)
    print(f"merges list = {merges_list}")
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
    # exit(0)
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    n_nodes = g.vcount()
    dendrogram(linkage_matrix, ax=ax3, labels=list(range(1, n_nodes+1)))
    plt.title("Girvan-Newman dendrogram")
    plt.xlabel("nodes")
    plt.ylabel("merge step")
    plt.tight_layout()
    plt.show()