# This script is very similar to time_analysis.py, but now for modularity instead of runtime

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Load data
# --------------------------------------------------

# Example:
with open("out/results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)



### analyse the average # edges, # comps per graph_size. 
# if avg deg, # comps the same per graph size, just report aggregate number.


no_edges_per_graph_size_df = {} # key = graphsize
no_edges_per_graph_size_run_df = {} # key = graphsize,run
no_concomps_per_graph_size_df = {} # key = graphsize
no_concomps_per_graph_size_run_df = {} # key = graphsize,run

for _, row in df.iterrows():
    if not isinstance(row["all_numbers_of_edges"], list):
        continue
    list_no_edges = row["all_numbers_of_edges"]
    list_no_concomps = row["all_numbers_of_concomps"]
    graph_size = row["graph_size"]
    run = row["run"]

    # group no_edges,no_concomps per (dendrogram (run), graph_size), list mean deg per run if interesting diff
    if (run, graph_size) in no_edges_per_graph_size_run_df:
        no_edges_per_graph_size_run_df[(run, graph_size)].extend(list_no_edges)
        no_concomps_per_graph_size_run_df[(run, graph_size)].extend(list_no_concomps)
    else:
        no_edges_per_graph_size_run_df[(run, graph_size)] = list(list_no_edges)
        no_concomps_per_graph_size_run_df[(run, graph_size)] = list(list_no_concomps)

    # group no_edges,no_concomps per (graph_size), display boxplots of stats 
    if (graph_size) in no_edges_per_graph_size_df:
        no_edges_per_graph_size_df[graph_size].extend(list_no_edges)
        no_concomps_per_graph_size_df[graph_size].extend(list_no_concomps)
    else:
        no_edges_per_graph_size_df[graph_size] = list(list_no_edges)
        no_concomps_per_graph_size_df[graph_size] = list(list_no_concomps)

# collect mean degree per run per graph_size, then report mean ± std across runs
from collections import defaultdict
mean_deg_per_graph_size = defaultdict(list)
mean_concomps_per_graph_size = defaultdict(list)
for (run, graph_size), vals in no_edges_per_graph_size_run_df.items():
    mean_deg_per_graph_size[graph_size].append(sum(vals) / len(vals) / graph_size)
for (run, graph_size), vals in no_concomps_per_graph_size_run_df.items():
    mean_concomps_per_graph_size[graph_size].append(sum(vals) / len(vals))

print("Mean degree per graph_size (mean ± std across runs):")
for graph_size in sorted(mean_deg_per_graph_size):
    run_means = mean_deg_per_graph_size[graph_size]
    m = sum(run_means) / len(run_means)
    s = (sum((v - m)**2 for v in run_means) / len(run_means))**0.5
    print(f"  graph_size={graph_size}: mean deg = {m:.4f} ± {s:.4f}")

print("\nMean connected components per graph_size (mean ± std across runs):")
for graph_size in sorted(mean_concomps_per_graph_size):
    run_means = mean_concomps_per_graph_size[graph_size]
    m = sum(run_means) / len(run_means)
    s = (sum((v - m)**2 for v in run_means) / len(run_means))**0.5
    print(f"  graph_size={graph_size}: mean concomps = {m:.4f} ± {s:.4f}")

# boxplot of no_edges per graph_size
pandas_edges_df = pd.DataFrame([
    {"graph_size": graph_size, "no_edges": v}
    for graph_size, vals in no_edges_per_graph_size_df.items()
    for v in vals
])

plt.figure(figsize=(10, 5))
sns.boxplot(data=pandas_edges_df, x="graph_size", y="no_edges")
plt.xlabel("Graph size")
plt.ylabel("Number of edges")
plt.title("Edge count distribution per graph size")
plt.tight_layout()
plt.savefig("figures/no_edges.png")

# boxplot of no_concomps per graph_size
pandas_concomps_df = pd.DataFrame([
    {"graph_size": graph_size, "no_concomps": v}
    for graph_size, vals in no_concomps_per_graph_size_df.items()
    for v in vals
])

plt.figure(figsize=(10, 5))
sns.boxplot(data=pandas_concomps_df, x="graph_size", y="no_concomps")
plt.xlabel("Graph size")
plt.ylabel("Number of connected components")
plt.title("Connected components distribution per graph size")
plt.tight_layout()
plt.savefig("figures/no_concomps.png")


