import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Spearman's rank correlation is used here rather than Pearson's because we care
# about monotonic ordering — does an algorithm that ranks higher in modularity
# also tend to rank higher in nHMI? — not the linear magnitude of differences.
# This makes ρ robust to scale discrepancies between the two metrics and to
# outliers, and directly captures the pairwise concordance described below.

results_path = Path(__file__).parent / "out" / "results.json"

with open(results_path) as f:
    all_data = json.load(f)

from itertools import combinations

N_GRAPHS_VALS = [100]

# Index all_data by (alg, graph_size, run) -> {n_graphs: entry}
entry_index: dict[tuple, dict] = defaultdict(dict)
for e in all_data:
    key = (e["algorithm"], e["graph_size"], e["run"])
    entry_index[key][e["n_graphs"]] = e

# vec{A, n, i} = [avg_mod@10, avg_mod@20, avg_mod@50, avg_mod@100,
#                 nhmi@10,    nhmi@20,    nhmi@50,    nhmi@100]
# Concatenating both metrics across all n_graphs gives an 8-element profile.
# spearmanr(vec_A, vec_B) then measures how consistently algorithm A ranks
# relative to B across every (metric, n_graphs) combination — high rho means
# A dominates (or is dominated by) B in the same way for both modularity and nHMI.
vectors: dict[tuple, np.ndarray] = {}
for (alg, gs, run), ng_dict in entry_index.items():
    mod_val  = ng_dict[100]["avg_modularity"]
    nhmi_val = ng_dict[10]["nhmi_score"]
    vectors[(alg, gs, run)] = np.array([mod_val, nhmi_val])

graph_sizes = sorted({gs  for _, gs, _   in vectors})
algorithms  = sorted({alg for alg, _, _  in vectors})
runs        = sorted({run for _, _, run  in vectors})

# For each (n, i): average Spearman rho over all C(6,2)=15 algorithm pairs.
# Then average those per-run values over all runs to get one number per graph size.
size_run_rhos: dict[int, list[float]] = defaultdict(list)

for gs in graph_sizes:
    for run in runs:
        pair_rhos = [
            spearmanr(vectors[(a, gs, run)], vectors[(b, gs, run)]).statistic
            for a, b in combinations(algorithms, 2)
            if (a, gs, run) in vectors and (b, gs, run) in vectors
        ]
        if pair_rhos:
            print(np.mean(pair_rhos))
            size_run_rhos[gs].append(np.mean(pair_rhos))

print(f"{'graph_size':>12}  {'mean rho':>9}  {'std rho':>8}  {'n_runs':>7}")
print("-" * 44)
for gs in graph_sizes:
    rhos = size_run_rhos[gs]
    print(f"{gs:>12}  {np.mean(rhos):>9.4f}  {np.std(rhos):>8.4f}  {len(rhos):>7}")

# --- plots ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. Scatter: avg_modularity vs nhmi_score at n_graphs=100, coloured by algorithm
data_100 = [e for e in all_data if e["n_graphs"] == 100]
alg_colors = dict(zip(algorithms, plt.cm.tab10(np.linspace(0, 1, len(algorithms)))))

ax = axes[0]
for e in data_100:
    ax.scatter(e["avg_modularity"], e["nhmi_score"],
               color=alg_colors[e["algorithm"]], alpha=0.5, s=20)
handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=alg_colors[a], markersize=7, label=a)
           for a in algorithms]
ax.legend(handles=handles, fontsize=8)
ax.set_xlabel("avg modularity")
ax.set_ylabel("nHMI score")
ax.set_title("Modularity vs nHMI  (n_graphs=100)")

# 2. Box plot of per-run average rho by graph size
ax = axes[1]
ax.boxplot([size_run_rhos[gs] for gs in graph_sizes],
           tick_labels=[str(gs) for gs in graph_sizes])
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("graph size")
ax.set_ylabel("mean pairwise Spearman rho  (per run)")
ax.set_title("Algorithm profile similarity by graph size")

plt.tight_layout()
out_path = Path(__file__).parent / "out" / "compare_mod_nhmi.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
