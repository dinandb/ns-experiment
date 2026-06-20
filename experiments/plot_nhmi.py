import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_json('out/results.json')

# nhmi graph
plt.figure(figsize=(12,6))
plt.title(f"nHMI score for all algorithms")
plt.xlabel("n_graphs")
plt.ylabel("nHMI score")

algorithm_order = [
    "GirvanNewman",
    "ClusteringCoefficientDivisive",
    "EdgeDegreeCentrality",
    "Distance",
    "CosineSimilarity",
    "Spectral",
]
for algorithm, data in sorted(df[df["graph_size"] == 500].groupby("algorithm"), key=lambda x: algorithm_order.index(x[0])):
    x = []
    y = []
    for n_graph, data_per_n_graphs in data.groupby("n_graphs"):
        nhmi_scores = data_per_n_graphs["nhmi_score"]
        median = nhmi_scores.median()
        x.append(n_graph)
        y.append(median)

    plt.plot(x, y, label=f"{algorithm}")

plt.legend(loc="upper left")
plt.savefig('figures/nhmi_graph.png')
plt.close()


#nhmi boxplot
plt.figure(figsize=(12,6))
sns.boxplot(
        data=df[df["graph_size"] == 500],
        x="n_graphs",
        y="nhmi_score",
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
plt.xlabel("n_graphs")
plt.ylabel("nHMI score")
plt.title("nHMi score per n_graphs")
plt.tight_layout()
plt.savefig('figures/nhmi_boxplot.png')
plt.close()