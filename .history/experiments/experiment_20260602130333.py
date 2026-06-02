import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyhrg.hrg import Dendrogram
from algorithms.divisive.girvan_newman import GirvanNewman
from algorithms.divisive.fast_GN import FastGirvanNewman
from algorithms.divisive.clust_coef_divisive import ClusteringCoefficientDivisive
from algorithms.agglomerative.cosine_similarity import CosineSimilarity
from algorithms.agglomerative.linkage import Linkage
from superalgorithm import SuperAlgorithm
from HMI import nhmi
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import dendrogram_generator
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage as scipy_linkage, cophenet
from scipy.spatial.distance import squareform
import time
import random as rnd
import networkx as nx
import json


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


def _merges_to_linkage(merges: list[tuple[int, int]], n: int) -> np.ndarray:
    all_children = {node for pair in merges for node in pair}
    sizes = {i: 1 for i in range(n)}
    rows = []

    for step, (a, b) in enumerate(merges):
        size = sizes.get(a, 1) + sizes.get(b, 1)
        rows.append([float(a), float(b), float(step + 1), float(size)])
        sizes[n + step] = size

    roots = [i for i in range(n) if i not in all_children]
    roots += [n + step for step in range(len(merges)) if n + step not in all_children]

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


def run_and_compare(n_nodes: int, dend: Dendrogram, n_graphs: int, algorithm) -> tuple[float, float]:
    """Run 'algorithm' on 'n_graphs' sampled graphs and compare the average dendrogram to the ground-truth dendrogram.

    Returns the nHMI score between the ground-truth merges and the algorithm's averaged dendrogram.
    """

    M = dendrogram_to_merges(dend)
    coph_sum = np.zeros((n_nodes, n_nodes))
    optimal_modularities = []
    for _ in range(n_graphs):
        g_nx = dend.generate_graph()
        g = ig.Graph.from_networkx(g_nx)
    
        M_prime = algorithm.run(g)
        optimal_modularity = SuperAlgorithm.optimize_for_modularity(g, M_prime)[1]
        optimal_modularities.append(optimal_modularity)

        lm = _merges_to_linkage(M_prime, n_nodes)
        coph_sum += squareform(cophenet(lm))

    avg_coph = coph_sum / n_graphs
    condensed = squareform(avg_coph)
    avg_lm = scipy_linkage(condensed, method="average")
    avg_M = [(int(row[0]), int(row[1])) for row in avg_lm]
    nhmi_score = nhmi(n_nodes, M, avg_M, verbose=False)
    avg_modularity = float(np.mean(optimal_modularities))
    return nhmi_score, avg_modularity

GRAPH_SIZES = [20, 50, 100]#, 500, 1000]
N_RUNS = 2
N_GRAPHS_LIST = [10, 20]#, 50, 100]
ALGORITHMS = [GirvanNewman(), FastGirvanNewman(), ClusteringCoefficientDivisive(), CosineSimilarity(), Linkage()]

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')


def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for graph_size in GRAPH_SIZES:
        for run_i in range(N_RUNS):
            dend = dendrogram_generator.make_rnd_dendrogram(n=graph_size)
            print(f"graph_size={graph_size}, run={run_i+1}/{N_RUNS}")

            for alg in ALGORITHMS:
                alg_name = alg.__class__.__name__
                for n_graphs in N_GRAPHS_LIST:
                    t0 = time.perf_counter()
                    nhmi_score, avg_modularity = run_and_compare(graph_size, dend, n_graphs, alg)
                    elapsed = time.perf_counter() - t0

                    record = {
                        "graph_size": graph_size,
                        "run": run_i,
                        "algorithm": alg_name,
                        "n_graphs": n_graphs,
                        "nhmi_score": nhmi_score,
                        "avg_modularity": avg_modularity,
                        "time_s": elapsed,
                    }
                    results.append(record)
                    print(f"  {alg_name} n_graphs={n_graphs}: nhmi={nhmi_score:.4f}, mod={avg_modularity:.4f} ({elapsed:.2f}s)")

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    run_experiment()

#   we want to see if opt for mod scores relatively coincide (and to what extent) with nhmi_scores.
# i.e. if we see that nhmi_score for alg A is lower than nhmi_score for alg B how often do we also see in the same instance that opt for mod score for alg A is also lower than opt for mod score for alg B



# check above code

# claude recommendations for final comments:
#   The natural metric is Spearman's rank correlation between the per-instance nhmi rankings and
#   modularity rankings of algorithms. For each (dend, n_graphs) instance you have a vector of
#   nhmi scores across algorithms and a vector of modularity scores — Spearman's ρ tells you how
#   consistently the two orderings agree, averaged across instances.

# maybe plot this Spearman's ρ dependent on n_graphs. 

#   Concretely, after the experiment you'd group results.json by (graph_size, run, n_graphs),
#   rank algorithms by nhmi and by modularity within each group, and compute
#   scipy.stats.spearmanr. A ρ near 1 means modularity rank reliably predicts nhmi rank.

#   Two options for where to put this analysis:
#   - Inline in a second script (experiments/analyze.py) that reads results.json — keeps
#   experiment and analysis separate, easier to re-run analysis without re-running the
#   experiment.
#   - At the bottom of run_experiment() — convenient but couples slow data collection to fast
#   analysis.

#   I'd recommend a separate analyze.py given experiments are expensive. Want me to implement
#   that now?