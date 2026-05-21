import math


def build_tree(n_nodes, merges):
    children = {}
    leaves = {i: frozenset([i]) for i in range(n_nodes)}
    for idx, (a, b) in enumerate(merges):
        cid = n_nodes + idx
        children[cid] = (a, b)
        leaves[cid] = leaves[a] | leaves[b]
    return children, leaves


def _H(delta, v_inter, leaves, verbose=False, indent="", label=""):
    """H(delta_v | v inter v') -- eq. 5."""
    n = len(v_inter)
    if n == 0:
        return 0.0
    total = 0.0
    terms = []
    for u in delta:
        k = len(leaves[u] & v_inter)
        if k > 0:
            p = k / n
            contrib = -p * math.log(p)
            total += contrib
            terms.append((u, k, p, contrib))
    if verbose:
        print(f"{indent}  H{label}(delta | v_inter={sorted(v_inter)}), n={n}:")
        for u, k, p, contrib in terms:
            print(f"{indent}    u={u}: |leaves(u) & v_inter|={k}, p={k}/{n}={p:.4f}, -p*ln(p)={contrib:.6f}")
        print(f"{indent}    H{label} = {total:.6f}")
    return total


def _H_joint(delta1, delta2, n, leaves1, leaves2, verbose=False, indent=""):
    """H(delta_v inter delta_v' | v inter v') -- eq. 6."""
    if n == 0:
        return 0.0
    total = 0.0
    terms = []
    for u in delta1:
        for u_prime in delta2:
            k = len(leaves1[u] & leaves2[u_prime])
            if k > 0:
                p = k / n
                contrib = -p * math.log(p)
                total += contrib
                terms.append((u, u_prime, k, p, contrib))
    if verbose:
        print(f"{indent}  H_joint(delta inter delta' | v_inter), n={n}:")
        for u, u_prime, k, p, contrib in terms:
            print(f"{indent}    (u={u}, u'={u_prime}): k={k}, p={k}/{n}={p:.4f}, -p*ln(p)={contrib:.6f}")
        print(f"{indent}    H_joint = {total:.6f}")
    return total


def _one_step_mi(delta1, delta2, v_inter, leaves1, leaves2, verbose=False, indent=""):
    """I(delta_v ; delta_v' | v inter v') -- eq. 4."""
    n = len(v_inter)
    if n == 0:
        return 0.0
    h1  = _H(delta1, v_inter, leaves1, verbose=verbose, indent=indent, label="1")
    h2  = _H(delta2, v_inter, leaves2, verbose=verbose, indent=indent, label="2")
    h12 = _H_joint(delta1, delta2, n, leaves1, leaves2, verbose=verbose, indent=indent)
    mi = h1 + h2 - h12
    if verbose:
        print(f"{indent}  one-step MI = H1 + H2 - H_joint = {h1:.6f} + {h2:.6f} - {h12:.6f} = {mi:.6f}")
    return mi


def _hmi_node(v, v_prime, children1, children2, leaves1, leaves2, verbose=False, depth=0):
    """Recursive HMI for a pair of nodes -- eq. 3."""
    indent = "  " * depth
    v_inter = leaves1[v] & leaves2[v_prime]

    if verbose:
        print(f"{indent}hmi_node(v={v}, v'={v_prime})")
        print(f"{indent}  leaves(v)={sorted(leaves1[v])}, leaves(v')={sorted(leaves2[v_prime])}")
        print(f"{indent}  v_inter={sorted(v_inter)}, |v_inter|={len(v_inter)}")

    if v not in children1 or v_prime not in children2:
        if verbose:
            print(f"{indent}  leaf node -> 0")
        return 0.0

    delta1 = list(children1[v])
    delta2 = list(children2[v_prime])

    if verbose:
        print(f"{indent}  delta_v={delta1}, delta_v'={delta2}")

    result = _one_step_mi(delta1, delta2, v_inter, leaves1, leaves2, verbose=verbose, indent=indent)

    n = len(v_inter)
    if n > 0:
        for u in delta1:
            for u_prime in delta2:
                k = len(leaves1[u] & leaves2[u_prime])
                if k > 0:
                    weight = k / n
                    sub = _hmi_node(u, u_prime, children1, children2, leaves1, leaves2,
                                    verbose=verbose, depth=depth + 1)
                    if verbose:
                        print(f"{indent}  recursive contrib: u={u}, u'={u_prime}, "
                              f"k={k}, weight={weight:.4f}, sub={sub:.6f}, "
                              f"weight*sub={weight*sub:.6f}")
                    result += weight * sub

    if verbose:
        print(f"{indent}  => hmi_node({v},{v_prime}) = {result:.6f}")

    return result


def hmi(n_nodes, merges1, merges2, verbose=False):
    """
    Hierarchical Mutual Information between two dendrograms.

    Parameters
    ----------
    n_nodes : int
        Number of leaf nodes, shared by both dendrograms.
    merges1, merges2 : list of (int, int)
        Agglomerative merge sequences.  Merge k produces cluster
        n_nodes + k; the root is n_nodes + len(merges) - 1.
    verbose : bool
        Print step-by-step computation of every entropy and MI term.

    Returns
    -------
    float
        HMI value (nats).
    """
    children1, leaves1 = build_tree(n_nodes, merges1)
    children2, leaves2 = build_tree(n_nodes, merges2)

    root1 = n_nodes + len(merges1) - 1
    root2 = n_nodes + len(merges2) - 1

    if verbose:
        print("=== HMI computation ===")
        print(f"n_nodes={n_nodes}, root1={root1}, root2={root2}")
        print(f"merges1={merges1}")
        print(f"merges2={merges2}")
        print()

    result = _hmi_node(root1, root2, children1, children2, leaves1, leaves2,
                       verbose=verbose, depth=0)

    if verbose:
        print(f"\n=== HMI = {result:.6f} nats ===")

    return result


def nhmi(n_nodes, merges1, merges2, verbose=False):
    """
    Normalized Hierarchical Mutual Information — eq. formulas3.

        i(T; T') = I(T; T') / sqrt( I(T; T) * I(T'; T') )

    Returns a value in [0, 1].  Returns 0.0 if either self-information is 0
    (a degenerate tree with no internal structure).

    Parameters
    ----------
    n_nodes : int
        Number of leaf nodes, shared by both dendrograms.
    merges1, merges2 : list of (int, int)
        Agglomerative merge sequences.
    verbose : bool
        Print each of the three HMI computations in full.

    Returns
    -------
    float
    """
    if verbose:
        print("=== nHMI: computing I(T; T') ===")
    i_12 = hmi(n_nodes, merges1, merges2, verbose=verbose)

    if verbose:
        print("\n=== nHMI: computing I(T; T) ===")
    i_11 = hmi(n_nodes, merges1, merges1, verbose=verbose)

    if verbose:
        print("\n=== nHMI: computing I(T'; T') ===")
    i_22 = hmi(n_nodes, merges2, merges2, verbose=verbose)

    denom = math.sqrt(i_11 * i_22)
    if denom == 0.0:
        return 0.0

    result = i_12 / denom

    if verbose:
        print(f"\n=== nHMI = {i_12:.6f} / sqrt({i_11:.6f} * {i_22:.6f})"
              f" = {i_12:.6f} / {denom:.6f} = {result:.6f} ===")

    return result
