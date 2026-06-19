import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_json('experiments/out/results.json')

# nhmi graph
plt.figure(figsize=(12,6))
plt.title(f"nHMI score for all algorithms")
plt.xlabel("Graph size")
plt.ylabel("nHMI score")

for algorithm, data in df.groupby("algorithm"):
    x = []
    y = []
    for n_graph, data_per_n_graphs in data.groupby("n_graphs"):
        nhmi_scores = data_per_n_graphs["nhmi_score"]
        median = nhmi_scores.median()
        x.append(n_graph)
        y.append(median)

    plt.plot(x, y, label=f"{algorithm}")

plt.legend(loc="upper left")
plt.savefig('experiments/figures/nhmi_graph.png')
plt.close()


#nhmi boxplot
plt.figure(figsize=(12,6))
sns.boxplot(
        data=df,
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
plt.xlabel("Graph size")
plt.ylabel("nHMI score")
plt.title("nHMi score per graph size")
plt.tight_layout()
plt.savefig('experiments/figures/nhmi_boxplot.png')
plt.close()