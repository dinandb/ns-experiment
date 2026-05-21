import igraph as ig
import dendrogram_generator
from HMI import nhmi
from algorithms.divisive.girvan_newman import GirvanNewman
from algorithms.divisive.clust_coef_divisive import ClusteringCoefficientDivisive


DIVISIVE_ALGORITHMS = [GirvanNewman(), ClusteringCoefficientDivisive()]
AGGLOMERATIVE_ALGORITHMS = []
ALL_ALGORITHMS = DIVISIVE_ALGORITHMS + AGGLOMERATIVE_ALGORITHMS


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


def run_and_compare(n_dendrograms: int, n_graphs_per_dendrogram: int, n_nodes: int = 20):
    """Run all algorithms on sampled graphs and compare their merges to the ground-truth dendrogram.

    Args:
        n_dendrograms: number of random dendrograms to generate
        n_graphs_per_dendrogram: number of graphs sampled per dendrogram
        n_nodes: number of leaf nodes per dendrogram
    """
    algo_names = [algo.__class__.__name__ for algo in ALL_ALGORITHMS]
    total_scores = {name: 0.0 for name in algo_names}

    for dend_i in range(n_dendrograms):
        dend = dendrogram_generator.make_rnd_dendrogram(n_nodes)
        M = dendrogram_to_merges(dend)

        dend_scores = {name: 0.0 for name in algo_names}

        for graph_j in range(n_graphs_per_dendrogram):
            g_nx = dend.generate_graph()
            g = ig.Graph.from_networkx(g_nx)

            for algo in ALL_ALGORITHMS:
                name = algo.__class__.__name__
                M_prime = algo.run(g)
                score = nhmi(n_nodes, M, M_prime, verbose=False)
                dend_scores[name] += score

        for name in algo_names:
            avg = dend_scores[name] / n_graphs_per_dendrogram
            total_scores[name] += avg
            print(f"  Dendrogram {dend_i + 1}, {name}: avg nHMI = {avg:.4f}")

    print("\n=== Overall average nHMI per algorithm ===")
    for name in algo_names:
        print(f"  {name}: {total_scores[name] / n_dendrograms:.4f}")


if __name__ == "__main__":
    run_and_compare(n_dendrograms=5, n_graphs_per_dendrogram=10, n_nodes=10)
