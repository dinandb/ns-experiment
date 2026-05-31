import igraph as ig

from algorithms.divisive.girvan_newman import run as gn_run
from algorithms.divisive.clust_coef_divisive import run as cc_run
from algorithms.agglomerative.linkage import run as ag_run
from superalgorithm import SuperAlgorithm
from HMI import hmi, nhmi


def dendrogram_to_merges(dendrogram) -> list[tuple[int, int]]:
    """Convert a pyhrg Dendrogram to an agglomerative merges list for nhmi.

    Post-order traversal ensures children are assigned IDs before their parent.
    Merge k (0-based) produces internal node n+k, matching HMI's convention.
    """
    leaf_nodes = [v for v in dendrogram.nodes() if not dendrogram.is_dendrogram_node(v)]
    n = len(leaf_nodes)

    dnode_to_merge_idx = {}
    merges = []

    def postorder(node):
        left = dendrogram.nodes[node]["left"]
        right = dendrogram.nodes[node]["right"]
        if dendrogram.is_dendrogram_node(left):
            postorder(left)
        if dendrogram.is_dendrogram_node(right):
            postorder(right)
        a = left if not dendrogram.is_dendrogram_node(left) else n + dnode_to_merge_idx[left]
        b = right if not dendrogram.is_dendrogram_node(right) else n + dnode_to_merge_idx[right]
        merges.append((a, b))
        dnode_to_merge_idx[node] = len(merges) - 1

    postorder("_D0")
    return merges

# Simple 8-node graph with clear two-level hierarchical structure:
#
#   Community A (nodes 0-3)        Community B (nodes 4-7)
#   ┌─────────────────┐            ┌─────────────────┐
#   │  0 ── 1          │            │  4 ── 5          │
#   │   \  /           │            │   \  /           │
#   │    2 ── 3 ──────────────────── 4    6 ── 7      │
#   └─────────────────┘            └─────────────────┘
#
# Dense triangles (0-1-2) and (4-5-6) form the two sub-communities.
# Node 3 is weakly attached to A; node 7 weakly attached to B.
# Single bridge edge 3-4 connects the two communities.

edges = [
    (0, 1), (0, 2), (1, 2),   # tight triangle A
    (2, 3),                    # 3 weakly attached to A
    (4, 5), (4, 6), (5, 6),   # tight triangle B
    (6, 7),                    # 7 weakly attached to B
    (3, 4),                    # bridge between communities
]

g = ig.Graph(n=8, edges=edges)

print("=== Girvan-Newman ===")
merges_gn = gn_run(g)
print(f"merges: {merges_gn}\n")



# # print("=== CC-Divisive ===")
# merges_cc = cc_run(g)
# # print(f"merges: {merges_cc}\n")

# print("=== Agglomerative Linkage ===")
# merges_ag = ag_run(g, 'average')
# print(f"merges: {merges_ag}\n")

# score = nhmi(g.vcount(), merges_gn, merges_cc, verbose=False)
# print(f"\nnHMI = {score:.6f}")
# import dendrogram_generator
# rnd_dendrogram = dendrogram_generator.make_rnd_dendrogram(8)
# merges_rnd = dendrogram_to_merges(rnd_dendrogram)
# print(f"\n=== Random dendrogram merges ===\nmerges: {merges_rnd}")

# score_rnd = nhmi(8, merges_gn, merges_rnd, verbose=False)
