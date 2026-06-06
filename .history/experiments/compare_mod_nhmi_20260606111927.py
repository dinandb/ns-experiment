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
n_graphs = 10

vectors: dict[tuple, np.ndarray] = {}
for (alg, gs, run), ng_dict in entry_index.items():
    mod_val  = ng_dict[n_graphs]["avg_modularity"]
    nhmi_val = ng_dict[n_graphs]["nhmi_score"]
    vectors[(alg, gs, run)] = np.array([mod_val, nhmi_val])

graph_sizes = sorted({gs  for _, gs, _   in vectors})
algorithms  = sorted({alg for alg, _, _  in vectors})
runs        = sorted({run for _, _, run  in vectors})

# For each (n, i): average Spearman rho over all C(6,2)=15 algorithm pairs.
# Then average those per-run values over all runs to get one number per graph size.
size_run_rhos: dict[int, list[float]] = defaultdict(list)

for gs in graph_sizes:

    for run in runs:
        available_algs = [
            alg for alg in algorithms
            if (alg, gs, run) in vectors
        ]

        if len(available_algs) >= 2:
            mods = [
                vectors[(alg, gs, run)][0]
                for alg in available_algs
            ]

            nhmis = [
                vectors[(alg, gs, run)][1]
                for alg in available_algs
            ]

            rho = spearmanr(mods, nhmis).statistic

            print(rho)
            size_run_rhos[gs].append(rho)

# kijken of gemiddelde nemen over runs goed gaat
print(f"{'graph_size':>12}  {'mean rho':>9}  {'std rho':>8}  {'n_runs':>7}")
print("-" * 44)
for gs in graph_sizes:
    rhos = size_run_rhos[gs]
    # print(rhos)
    print(f"{gs:>12}  {np.mean(rhos):>9.4f}  {np.std(rhos):>8.4f}  {len(rhos):>7}")

# --- plots ---

fig, ax = plt.subplots(figsize=(7, 5))

ax.boxplot([size_run_rhos[gs] for gs in graph_sizes],
           tick_labels=[str(gs) for gs in graph_sizes])
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("graph size")
ax.set_ylabel("mean pairwise Spearman rho  (per run)")
ax.set_title("Algorithm profile similarity by graph size, n_graphs = {n_graphs}")

plt.tight_layout()
out_path = Path(__file__).parent / "out" / "compare_mod_nhmi.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
