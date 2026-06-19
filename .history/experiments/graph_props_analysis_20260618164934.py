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
    if not "all_numbers_of_edges" in row:
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
    if (run, graph_size) in no_edges_per_graph_size_run_df:
        no_edges_per_graph_size_run_df[(run, graph_size)].extend(list_no_edges)
        no_concomps_per_graph_size_run_df[(run, graph_size)].extend(list_no_concomps)
    else:
        no_edges_per_graph_size_run_df[(run, graph_size)] = list(list_no_edges)
        no_concomps_per_graph_size_run_df[(run, graph_size)] = list(list_no_concomps)

    algorithm = row["algorithm"]
    graph_size = row["graph_size"]
    modularities = row["optimal_modularities"]
    if (algorithm, graph_size) in modularity_df:
        modularity_df[(algorithm, graph_size)].extend(modularities)
    else:
        modularity_df[(algorithm, graph_size)] = list(modularities)

pandas_mod_df = pd.DataFrame([
    {"algorithm": algorithm, "graph_size": graph_size, "modularity": m}
    for (algorithm, graph_size), mods in modularity_df.items()
    for m in mods
])


plt.figure(figsize=(12, 6))

sns.boxplot(
    data=pandas_mod_df,
    x="graph_size",
    y="modularity",
    hue="algorithm",
    hue_order=[
        "GirvanNewman",
        "ClusteringCoefficientDivisive",
        "EdgeDegreeCentrality",
        "Distance",
        "CosineSimilarity",
        "Spectral",
    ],
)

plt.ylim(0,1)
plt.xlabel("Graph size")
plt.ylabel("Modularity")
plt.title("Modularity distribution per graph size")
plt.tight_layout()
plt.savefig('figures/modularity.png')
# plt.show()