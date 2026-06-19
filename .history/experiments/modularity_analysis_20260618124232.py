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

# --------------------------------------------------
# Convert cumulative timing measurements into
# per-graph timing samples
# --------------------------------------------------
#
# N_GRAPHS_LIST = [10, 20, 50, 100]
#
# time_s corresponds to:
#   10 graphs
#   +10 graphs
#   +30 graphs
#   +50 graphs
#
# Therefore divide by:
#   10, 10, 30, 50
#
# Then duplicate each per-graph time according to
# the number of graphs represented so that medians
# and quartiles become weighted naturally.
# --------------------------------------------------

modularity_df = {}

for _, row in df.iterrows():
    
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
plt.show()