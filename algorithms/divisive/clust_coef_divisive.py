import igraph as ig
import heapq
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from superalgorithm import SuperAlgorithm


class ClusteringCoefficientDivisive(SuperAlgorithm):
    """Clustering coefficient divisive community detection algorithm."""

    def run(self, g_orig: ig.Graph) -> list[tuple[int, int]]:
        """Run clustering coefficient divisive and return agglomerative merges compatible with HMI."""
        return _run_clustering_coefficient_divisive(g_orig)


def _run_clustering_coefficient_divisive(g_orig: ig.Graph) -> list[tuple[int, int]]:
    """
    Run the clustering-coefficient divisive algorithm on g_orig and return
    agglomerative merges compatible with HMI.

    Works on an internal copy so the caller's graph is not modified.
    Returns a list of (a, b) pairs where leaf node IDs are 0..n-1.
    """
    g = g_orig.copy()
    n_nodes = g.vcount()

    node_cc = g.transitivity_local_undirected(mode="zero")
    node_deg = list(g.degree())
    # T(v) = CC(v) * d(v) * (d(v)-1) / 2  — recover integer triangle counts from CC
    node_tri = [
        int(round(node_cc[v] * node_deg[v] * (node_deg[v] - 1) / 2))
        if node_deg[v] >= 2 else 0
        for v in range(n_nodes)
    ]

    def _cc(tri, deg):
        return 2.0 * tri / (deg * (deg - 1)) if deg >= 2 else 0.0

    def score(u, v):
        # Analytically compute sum of CC deltas for {u,v} ∪ common if (u,v) were removed.
        # No graph mutations, no igraph calls — purely O(degree).
        common = set(g.neighbors(u)) & set(g.neighbors(v))
        k = len(common)
        delta = 0.0
        # Each common neighbor w loses 1 triangle; degree unchanged
        for w in common:
            delta -= 2.0 / (node_deg[w] * (node_deg[w] - 1)) if node_deg[w] >= 2 else 0.0
        # u and v each lose k triangles and 1 degree
        delta += _cc(node_tri[u] - k, node_deg[u] - 1) - node_cc[u]
        delta += _cc(node_tri[v] - k, node_deg[v] - 1) - node_cc[v]
        return delta

    ver = {}
    heap = []
    for u, v in g.get_edgelist():
        e = (min(u, v), max(u, v))
        ver[e] = 0
        heapq.heappush(heap, (-score(u, v), 0, e))

    comp = [0] * n_nodes
    comp_nodes = {0: set(range(n_nodes))}
    next_cid = 1
    splits = []

    while heap:
        while heap:
            neg_s, ver_stored, e = heapq.heappop(heap)
            if ver.get(e) == ver_stored:
                break
        else:
            break

        u, v = e
        common = set(g.neighbors(u)) & set(g.neighbors(v))
        affected = {u, v} | common
        stale = {(min(w, nb), max(w, nb)) for w in affected for nb in g.neighbors(w)} - {e}

        g.delete_edges([e])
        del ver[e]

        new_mem = g.connected_components().membership
        if new_mem[u] != new_mem[v]:
            old_cid = comp[u]
            cid_a, cid_b = next_cid, next_cid + 1
            next_cid += 2
            for node in range(g.vcount()):
                if comp[node] == old_cid:
                    comp[node] = cid_a if new_mem[node] == new_mem[u] else cid_b
            splits.append((old_cid, cid_a, cid_b))
            comp_nodes[cid_a] = {n for n in comp_nodes[old_cid] if new_mem[n] == new_mem[u]}
            comp_nodes[cid_b] = comp_nodes[old_cid] - comp_nodes[cid_a]
            del comp_nodes[old_cid]

        # Update triangle counts, degrees and CC analytically after actual removal
        k = len(common)
        for w in common:
            node_tri[w] -= 1
            node_cc[w] = _cc(node_tri[w], node_deg[w])
        node_tri[u] -= k; node_deg[u] -= 1; node_cc[u] = _cc(node_tri[u], node_deg[u])
        node_tri[v] -= k; node_deg[v] -= 1; node_cc[v] = _cc(node_tri[v], node_deg[v])

        for e2 in stale:
            if e2 in ver:
                ver[e2] += 1
                heapq.heappush(heap, (-score(*e2), ver[e2], e2))

    # Convert divisive splits to agglomerative format
    gn_idx = {}
    gn_size = {i: 1 for i in range(n_nodes)}
    gn_merges = []
    gn_cur = n_nodes

    for cid, nodes in comp_nodes.items():
        nodes_list = sorted(nodes)
        if len(nodes_list) == 1:
            gn_idx[cid] = nodes_list[0]
        else:
            a = nodes_list[0]
            for b in nodes_list[1:]:
                gn_merges.append((a, b))
                gn_size[gn_cur] = gn_size[a] + gn_size[b]
                a = gn_cur
                gn_cur += 1
            gn_idx[cid] = a

    for parent, ca, cb in reversed(splits):
        ia, ib = gn_idx[ca], gn_idx[cb]
        gn_merges.append((ia, ib))
        gn_size[gn_cur] = gn_size[ia] + gn_size[ib]
        gn_idx[parent] = gn_cur
        gn_cur += 1

    return gn_merges


def run(g_orig: ig.Graph) -> list[tuple[int, int]]:
    """Run clustering coefficient divisive algorithm. Wrapper for backward compatibility."""
    algo = ClusteringCoefficientDivisive()
    return algo.run(g_orig)


if __name__ == "__main__":
    g = ig.Graph.Famous("Zachary")

    # fig, ax = plt.subplots()
    # ig.plot(g, target=ax, vertex_label=list(range(g.vcount())), vertex_size=20)
    # plt.show()

    node_cc = g.transitivity_local_undirected(mode="zero")

    def score(u, v):
        common = set(g.neighbors(u)) & set(g.neighbors(v))
        affected = list({u, v} | common)
        g.delete_edges([(u, v)])
        cc_after = g.transitivity_local_undirected(vertices=affected, mode="zero")
        g.add_edge(u, v)
        return sum(cc_after[i] - node_cc[w] for i, w in enumerate(affected))

    ver = {}
    heap = []
    for u, v in g.get_edgelist():
        e = (min(u, v), max(u, v))
        ver[e] = 0
        heapq.heappush(heap, (-score(u, v), 0, e))

    print(f"Nodes: {g.vcount()}, Edges: {g.ecount()}")
    print(f"node_cc (first 10): {[round(c, 4) for c in node_cc[:10]]}")
    print(f"heap size: {len(heap)}")
    print("Top 10 edges by v_e:")
    for neg_s, v, e in heapq.nsmallest(10, heap):
        print(f"  edge {e}  v_e={-neg_s:.6f}")

    n_nodes = g.vcount()
    merges = []
    comp = [0] * n_nodes
    comp_nodes = {0: set(range(n_nodes))}
    next_cid = 1

    while heap:
        while heap:
            neg_s, ver_stored, e = heapq.heappop(heap)
            if ver.get(e) == ver_stored:
                break
        else:
            break

        u, v = e
        common = set(g.neighbors(u)) & set(g.neighbors(v))
        affected = {u, v} | common
        stale = {(min(w, nb), max(w, nb)) for w in affected for nb in g.neighbors(w)} - {e}

        g.delete_edges([e])
        del ver[e]

        print(f"removed edge {e}  v_e={-neg_s:.6f}")

        new_mem = g.connected_components().membership
        if new_mem[u] != new_mem[v]:
            print("the above caused a split")
            old_cid = comp[u]
            cid_a, cid_b = next_cid, next_cid + 1
            next_cid += 2
            for node in range(g.vcount()):
                if comp[node] == old_cid:
                    comp[node] = cid_a if new_mem[node] == new_mem[u] else cid_b
            merges.append((old_cid, cid_a, cid_b))
            comp_nodes[cid_a] = {n for n in comp_nodes[old_cid] if new_mem[n] == new_mem[u]}
            comp_nodes[cid_b] = comp_nodes[old_cid] - comp_nodes[cid_a]
            del comp_nodes[old_cid]

        new_cc = g.transitivity_local_undirected(vertices=list(affected), mode="zero")
        for i, w in enumerate(affected):
            node_cc[w] = new_cc[i]

        for e2 in stale:
            if e2 in ver:
                ver[e2] += 1
                heapq.heappush(heap, (-score(*e2), ver[e2], e2))

    print(f"comp nodes after while loop: {comp_nodes}")
    divisive_merges = merges
    gn_idx = {}
    gn_size = {i: 1 for i in range(n_nodes)}
    gn_merges = []
    gn_cur = n_nodes

    for cid, nodes in comp_nodes.items():
        nodes_list = sorted(nodes)
        if len(nodes_list) == 1:
            gn_idx[cid] = nodes_list[0]
        else:
            a = nodes_list[0]
            for b in nodes_list[1:]:
                gn_merges.append((a, b))
                gn_size[gn_cur] = gn_size[a] + gn_size[b]
                a = gn_cur; gn_cur += 1
            gn_idx[cid] = a

    for parent, ca, cb in reversed(divisive_merges):
        ia, ib = gn_idx[ca], gn_idx[cb]
        gn_merges.append((ia, ib))
        gn_size[gn_cur] = gn_size[ia] + gn_size[ib]
        gn_idx[parent] = gn_cur; gn_cur += 1

    merges = gn_merges

    print(f"\nSplits recorded: {len(divisive_merges)}")

    scipy_idx = {}
    cluster_size = {i: 1 for i in range(n_nodes)}
    linkage_rows = []
    cur = n_nodes

    for cid, nodes in comp_nodes.items():
        nodes_list = sorted(nodes)
        if len(nodes_list) == 1:
            scipy_idx[cid] = nodes_list[0]
        else:
            a = nodes_list[0]
            for b in nodes_list[1:]:
                linkage_rows.append([float(a), float(b), 0.0, float(cluster_size[a] + cluster_size[b])])
                cluster_size[cur] = cluster_size[a] + cluster_size[b]
                a = cur; cur += 1
            scipy_idx[cid] = a

    for step, (parent, ca, cb) in enumerate(reversed(divisive_merges), start=1):
        ia, ib = scipy_idx[ca], scipy_idx[cb]
        count = cluster_size[ia] + cluster_size[ib]
        linkage_rows.append([float(ia), float(ib), float(step), float(count)])
        cluster_size[cur] = count
        scipy_idx[parent] = cur; cur += 1

    linkage_matrix = np.array(linkage_rows, dtype=float)

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, ax=ax3, labels=list(range(n_nodes)))
    ax3.set_title("CC-divisive dendrogram")
    ax3.set_xlabel("nodes")
    ax3.set_ylabel("split step (reversed)")
    print(f"merges = {merges}")
    plt.tight_layout()
    plt.show()
