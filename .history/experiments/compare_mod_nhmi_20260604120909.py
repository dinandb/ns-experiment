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

data = [e for e in all_data if e["n_graphs"] == 100]

# (graph_size, run) -> algorithm -> entry
groups: dict[tuple, dict] = defaultdict(dict)
for e in data:
    groups[(e["graph_size"], e["run"])][e["algorithm"]] = e

graph_sizes = sorted({gs for gs, _ in groups})

# For each (graph_size, run): rank the algorithms by avg_modularity and nhmi_score,
# then compute ρ across those N=6 data points. ρ > 0 means algorithms that achieved
# higher modularity also tended to achieve higher nHMI on that dendrogram.
# per_size_rhos: dict[int, list[float]] = defaultdict(list)

# for (gs, run), alg_dict in groups.items():
#     mod_vals = [alg_dict[a]["avg_modularity"] for a in alg_dict]
#     nhmi_vals = [alg_dict[a]["nhmi_score"] for a in alg_dict]
#     rho, _ = spearmanr(mod_vals, nhmi_vals)
#     per_size_rhos[gs].append(rho)

# print(f"{'graph_size':>12}  {'mean rho':>9}  {'std rho':>8}  {'n_runs':>7}")
# print("-" * 44)
# for gs in graph_sizes:
#     rhos = per_size_rhos[gs]
#     print(f"{gs:>12}  {np.mean(rhos):>9.4f}  {np.std(rhos):>8.4f}  {len(rhos):>7}")

all_mod = [e["avg_modularity"] for e in data]
all_nhmi = [e["nhmi_score"] for e in data]
rho_overall, p_overall = spearmanr(all_mod, all_nhmi)
print(f"\nOverall  rho = {rho_overall:.4f}  (p = {p_overall:.2e},  n = {len(all_mod)})")

# --- plots ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# # 1. Scatter: avg_modularity vs nhmi_score, coloured by algorithm
# algorithms = sorted({e["algorithm"] for e in data})
# colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
# color_map = dict(zip(algorithms, colors))

# ax = axes[0]
# for e in data:
#     ax.scatter(e["avg_modularity"], e["nhmi_score"],
#                color=color_map[e["algorithm"]], alpha=0.4, s=18,
#                label=e["algorithm"])

# handles = [plt.Line2D([0], [0], marker="o", color="w",
#                       markerfacecolor=color_map[a], markersize=7, label=a)
#            for a in algorithms]
# ax.legend(handles=handles, fontsize=8)
# ax.set_xlabel("avg modularity")
# ax.set_ylabel("nHMI score")
# ax.set_title(f"Modularity vs nHMI  (n_graphs=100, overall rho={rho_overall:.3f})")

# 2. Box plot of per-run ρ by graph size
ax = axes[1]
# ax.boxplot([per_size_rhos[gs] for gs in graph_sizes],
        #    tick_labels=[str(gs) for gs in graph_sizes])
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("graph size")
ax.set_ylabel("Spearman ρ  (per run)")
ax.set_title("Modularity–nHMI rank correlation by graph size")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "out" / "compare_mod_nhmi.png", dpi=150)
plt.show()
