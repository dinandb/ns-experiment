import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import dendrogram_generator
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage as scipy_linkage, cophenet
from scipy.spatial.distance import squareform
from HMI import nhmi
from algorithms.divisive.girvan_newman import GirvanNewman
from algorithms.divisive.edge_deg_centrality import EdgeDegreeCentrality
from algorithms.divisive.clust_coef_divisive import ClusteringCoefficientDivisive
from algorithms.agglomerative.cosine_similarity import ClusteringCoefficientDivisive
from algorithms.agglomerative.distance import ClusteringCoefficientDivisive
from algorithms.agglomerative.clust_coef_divisive import ClusteringCoefficientDivisive


DIVISIVE_ALGORITHMS = [GirvanNewman(), ClusteringCoefficientDivisive(), EdgeDegreeCentrality()]
AGGLOMERATIVE_ALGORITHMS = []
ALL_ALGORITHMS = DIVISIVE_ALGORITHMS + AGGLOMERATIVE_ALGORITHMS


def _merges_to_linkage(merges: list[tuple[int, int]], n: int) -> np.ndarray:
    """Convert agglomerative merges list to scipy linkage matrix.

    If the merges don't span all n leaves (disconnected graph), dummy merges
    are appended at maximum distance to connect component roots.
    """
    all_children = {node for pair in merges for node in pair}
    sizes = {i: 1 for i in range(n)}
    rows = []

    for step, (a, b) in enumerate(merges):
        size = sizes.get(a, 1) + sizes.get(b, 1)
        rows.append([float(a), float(b), float(step + 1), float(size)])
        sizes[n + step] = size

    # Find roots: nodes never consumed as a child in any merge
    roots = [i for i in range(n) if i not in all_children]
    roots += [n + step for step in range(len(merges)) if n + step not in all_children]

    # Chain-merge component roots so scipy gets a complete tree
    next_id = n + len(merges)
    dummy_dist = float(len(merges) + 1)
    cur, cur_size = roots[0], sizes.get(roots[0], 1)
    for root in roots[1:]:
        root_size = sizes.get(root, 1)
        total = cur_size + root_size
        rows.append([float(cur), float(root), dummy_dist, float(total)])
        dummy_dist += 1.0
        sizes[next_id] = total
        cur, cur_size, next_id = next_id, total, next_id + 1

    return np.array(rows, dtype=float)


def visualize_graph(g: ig.Graph):
    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(g, target=ax, vertex_label=list(range(g.vcount())), vertex_label_size=8)
    ax.set_title(f"Graph  (n={g.vcount()} vertices, m={g.ecount()} edges)")
    plt.tight_layout()
    plt.show()


def visualize_merges(n: int, M: list[tuple[int, int]], M_prime: list[tuple[int, int]], algo_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Dendrogram comparison — {algo_name}")
    for ax, merges, title in zip(axes, [M, M_prime], ["Ground truth (M)", f"Algorithm (M')"]):
        linkage = _merges_to_linkage(merges, n)
        scipy_dendrogram(linkage, ax=ax, labels=list(range(n)), leaf_rotation=90, leaf_font_size=7)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def dendrogram_to_merges(dendrogram) -> list[tuple[int, int]]:
    """Convert a pyhrg Dendrogram to an agglomerative merges list for nhmi."""
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


def run_and_compare(n_dendrograms: int, n_graphs_per_dendrogram: int, n_nodes: int):
    """Run all algorithms on sampled graphs and compare their merges to the ground-truth dendrogram.

    Args:
        n_dendrograms: number of random dendrograms to generate
        n_graphs_per_dendrogram: number of graphs sampled per dendrogram
        n_nodes: number of leaf nodes per dendrogram
    """
    algo_names = [algo.__class__.__name__ for algo in ALL_ALGORITHMS]
    total_scores = {name: 0.0 for name in algo_names}
    total_algo_times = {name: 0.0 for name in algo_names}

    run_start = time.perf_counter()

    for dend_i in range(n_dendrograms):
        dend_start = time.perf_counter()
        dend = dendrogram_generator.make_rnd_dendrogram(n_nodes)
        t_make_dend = time.perf_counter() - dend_start

        t_merges_start = time.perf_counter()
        M = dendrogram_to_merges(dend)
        t_merges = time.perf_counter() - t_merges_start

        dend_algo_times = {name: 0.0 for name in algo_names}
        dend_coph_sums = {name: np.zeros((n_nodes, n_nodes)) for name in algo_names}

        for graph_j in range(n_graphs_per_dendrogram):
            g_nx = dend.generate_graph()
            g = ig.Graph.from_networkx(g_nx)
            # visualize_graph(g)

            for algo in ALL_ALGORITHMS:
                name = algo.__class__.__name__
                algo_start = time.perf_counter()
                M_prime = algo.run(g)
                # visualize_merges(n_nodes, M, M_prime, name)
                # exit(0)
                dend_algo_times[name] += time.perf_counter() - algo_start
                lm = _merges_to_linkage(M_prime, n_nodes)
                dend_coph_sums[name] += squareform(cophenet(lm))

        # average cophenetic matrix across all sampled graphs
        dend_elapsed = time.perf_counter() - dend_start
        print(f"Dendrogram {dend_i + 1}/{n_dendrograms} ({dend_elapsed:.2f}s, make={t_make_dend:.3f}s, to_merges={t_merges:.3f}s):")
        for name in algo_names:
            avg_coph = dend_coph_sums[name] / n_graphs_per_dendrogram
            condensed = squareform(avg_coph)
            avg_lm = scipy_linkage(condensed, method="average")
            avg_M = [(int(row[0]), int(row[1])) for row in avg_lm]
            # visualize_merges(n_nodes, M, avg_M, name)
            score = nhmi(n_nodes, M, avg_M, verbose=False)
            total_scores[name] += score
            total_algo_times[name] += dend_algo_times[name]
            print(f"  {name}: nHMI(avg dendrogram) = {score:.4f}  ({dend_algo_times[name]:.2f}s)")

    total_elapsed = time.perf_counter() - run_start
    print(f"\n=== Overall average nHMI(avg dendrogram) per algorithm (total wall time: {total_elapsed:.2f}s) ===")
    for name in algo_names:
        print(f"  {name}: {total_scores[name] / n_dendrograms:.4f}  (total algo time: {total_algo_times[name]:.2f}s)")


if __name__ == "__main__":
    run_and_compare(n_dendrograms=1, n_graphs_per_dendrogram=1, n_nodes=500)
