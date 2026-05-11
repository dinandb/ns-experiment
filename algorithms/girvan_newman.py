import igraph as ig
import networkx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# g_networkx = networkx.gnm_random_graph(500, 800)

g = ig.Graph.Erdos_Renyi(n=500, m=800).clusters().giant()

# g = ig.Graph(n=g_networkx.number_of_nodes(), edges=list(g_networkx.edges()))

clusters = g.community_edge_betweenness()
communities = clusters.as_clustering()

print(communities)

# Build scipy linkage matrix from igraph merges.
# igraph's merges may not be in topological order (parent before children),
# so we sort them first: a merge can only run once both its inputs exist.
n = g.vcount()
merges_list = [(int(a), int(b)) for a, b in clusters.merges]

print(merges_list)
exit(0)



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
            available.add(n + old_step)  # keep old IDs so future lookups match
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

plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, no_labels=True)
plt.title("Girvan-Newman dendrogram")
plt.xlabel("nodes")
plt.ylabel("merge step")
plt.tight_layout()
plt.show()