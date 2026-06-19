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
# key = graph size -> avg # edges, avg # comps

degrees_per_graph_size_df = {} # key = graphsize

for _, row in df.iterrows():
    if not row.keys().contains("all_numbers_of_edges"):
        continue
    # group no_edges,no_concomps per (dendrogram (run), graph_size), list mean deg per run if interesting diff
    

    # group no_edges,no_concomps per (graph_size), display boxplots of stats 


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