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

for each alg A, graph size n, run i -> save vector in a dict with key {A, n, i}.KeyboardInterrupt
    
for all graph size n, run i:
    for all pairs of algs A, B:
        spearmanscorr (vec{A,n,i}, vec{B,n,i})
