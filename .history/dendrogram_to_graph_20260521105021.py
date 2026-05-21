import random
import igraph as ig
from pyhrg.hrg import Dendrogram
import dendogram_generator


def _build_leaf_to_ancestors(dendrogram: Dendrogram) -> dict:
    """Map each leaf to its ordered list of ancestors from root to leaf."""
    ancestors = {}

    def dfs(node, path):
        if not dendrogram.is_dendrogram_node(node):
            ancestors[node] = path
            return
        left = dendrogram.nodes[node]["left"]
        right = dendrogram.nodes[node]["right"]
        dfs(left, path + [node])
        dfs(right, path + [node])

    dfs("_D0", [])
    return ancestors


def _lowest_common_ancestor(ancestors_i, ancestors_j):
    """Return the deepest shared ancestor between two ancestor paths."""
    set_j = set(ancestors_j)
    lca = None
    for node in reversed(ancestors_i):
        if node in set_j:
            lca = node
            break
    return lca


def dendrogram_to_graph(dendrogram: Dendrogram) -> ig.Graph:
    """Sample a random graph from a Hierarchical Random Graph dendrogram.

    For each pair of leaf nodes (i, j), the edge probability is the 'p'
    value of their lowest common ancestor internal node.
    """
    leaves = [v for v in dendrogram.nodes() if not dendrogram.is_dendrogram_node(v)]
    n = len(leaves)

    leaf_to_ancestors = _build_leaf_to_ancestors(dendrogram)

    edges = []
    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            i = leaves[idx_i]
            j = leaves[idx_j]
            lca = _lowest_common_ancestor(leaf_to_ancestors[i], leaf_to_ancestors[j])
            p = dendrogram.nodes[lca]["p"]
            if random.random() < p:
                edges.append((idx_i, idx_j))

    return ig.Graph(n=n, edges=edges)
dendrogram = dendogram_generator.make_rnd_dendrogram(10)
print(dendrogram_to_graph(dendro))