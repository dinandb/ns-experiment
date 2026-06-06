

# read out/results.json

# read, for each pair of algorithms A, B, graph_size n, run i:
    # look at the entries with n_graphs = 100
    # denote the entry of alg A with e_A
    # denote the entry of alg B with e_B
    # extract e_A["avg_modularity"], e_A["nhmi_score"]
    # extract e_B["avg_modularity"], e_B["nhmi_score"]
    # i want to figure out the relationship between avg_modularity and nhmi_score
    # in particular, I want to count the number of times 