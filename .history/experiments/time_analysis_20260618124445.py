# Prompt:
# I have data like this:
#   {
#     "graph_size": 20,
#     "run": 0,
#     "algorithm": "GirvanNewman",
#     "n_graphs": 10,
#     "nhmi_score": 0.5906447630733161,
#     "avg_modularity": 0.38365842839408726,
#     "time_s": 0.03885650000302121
#   },
# We have this with the following lists of parameters:
# GRAPH_SIZES = [20, 50, 100, 500]
# N_RUNS = 10
# N_GRAPHS_LIST = [10, 20, 50, 100]
# ALGORITHMS = [GirvanNewman(), EdgeDegreeCentrality(), ClusteringCoefficientDivisive(), CosineSimilarity(), Distance(), Spectral()]

# For each algorithm, we want to group the runs based graph sizes on graph_sizes
# Then we set median points of the time data over the differet runs and n_graphs, together with first and third quartile points.
# Note that time_s is the elapsed time between adding more graphs to get to the next N_GRAPHS_LIST, so for adding 10, 10, 30, 50 graphs.
# So, we will need to divide time_s by these numbers to get the average time needed for running on a single graph.
# To mitigate this, we can add weights or duplicate the number 50 times in order to get the median.
# The second option is preffered, so we can use standard functions and libraries.
# Finally, we want to add lines between the median points to show the increase/decrease of time needed as the graphs get bigger.

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



timings_df = {}

for _, row in df.iterrows():


    algorithm = row["algorithm"]
    graph_size = row["graph_size"]
    timings = row["timings"]
    if (algorithm, graph_size) in timings_df:
        timings_df[(algorithm, graph_size)].extend(timings)
    else:
        timings_df[(algorithm, graph_size)] = list(timings)

pandas_timings_df = pd.DataFrame([
    {"algorithm": algorithm, "graph_size": graph_size, "timings": m}
    for (algorithm, graph_size), mods in timings_df.items()
    for m in mods
])


#     per_graph_time = row["time_s"] / n_added

#     # duplicate value n_added times
#     for _ in range(n_added):
#         expanded_rows.append(
#             {
#                 "algorithm": row["algorithm"],
#                 "graph_size": row["graph_size"],
#                 "per_graph_time": per_graph_time,
#             }
#         )

# expanded_df = pd.DataFrame(expanded_rows)

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=pandas_timings_df,
    x="graph_size",
    y="per_graph_time",
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

plt.yscale("log")
plt.xlabel("Graph size")
plt.ylabel("Time per graph (s)")
plt.title("Runtime distribution per graph size")
plt.tight_layout()
plt.savefig('figures/runtime.png')