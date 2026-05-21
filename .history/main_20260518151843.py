import igraph as ig

from algorithms.girvan_newman import run as gn_run
from algorithms.clust_coef_divisive import run as cc_run
from HMI import hmi, nhmi

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

# print("=== CC-Divisive ===")
# merges_cc = cc_run(g)
# print(f"merges: {merges_cc}\n")

# score = nhmi(g.vcount(), merges_gn, merges_cc, verbose=True)
# print(f"\nnHMI = {score:.6f}")
import dendogram_generator
print(dendogram_generator.make_rnd_dendrogram(10))


# we hebben een dendrogram -> merge list nodig.

# dan kunnen we de hmi van resultaat met ground truth vergelijken, want hmi neemt nu een merge list.

# maar beter zou zijn als de hmi gewoon een dendrogram neemt, dan hebben we juist een merge -> list (output algoritme) 