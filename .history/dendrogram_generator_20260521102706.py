from pyhrg.hrg import Dendrogram

import random as rnd
import networkx as nx


def generate_probs(n: int) -> list[float]:
    """
    Generate n random probabilities sorted ascending.
    """
    probs = [rnd.uniform(0.0, 1.0) for _ in range(n)]
    probs.sort()
    return probs


def assign_probs_topdown(
    dendrogram: Dendrogram,
    probs: list[float],
    node="_D0",
    index=0
) -> int:
    """
    Assign probabilities recursively in preorder traversal.

    Parents are assigned before children, guaranteeing:

        child.p >= parent.p
    """

    dendrogram.nodes[node]["p"] = probs[index]
    index += 1

    children = dendrogram.children(node)

    left_first = rnd.choice([True, False])
    first_child = dendrogram.nodes[node]["left"] if left_first else dendrogram.nodes[node]["right"]
    second_child = dendrogram.nodes[node]["left"] if not left_first else dendrogram.nodes[node]["right"]

    if dendrogram.is_dendrogram_node(first_child):
        index = assign_probs_topdown(
            dendrogram,
            probs,
            first_child,
            index
        )

    if dendrogram.is_dendrogram_node(second_child):
        index = assign_probs_topdown(
            dendrogram,
            probs,
            second_child,
            index
        )

    return index


def make_rnd_dendrogram(n: int) -> Dendrogram:

    # dummy graph with n vertices
    dummy_graph = nx.Graph()
    dummy_graph.add_nodes_from(range(n))

    # build random dendrogram structure
    dendrogram = Dendrogram(dummy_graph)
    dendrogram.initialize()

    # number of internal dendrogram nodes
    dnodes = list(dendrogram.dendrogram_nodes())
    num_dnodes = len(dnodes)

    # generate distributed probabilities
    probs = generate_probs(num_dnodes)

    # assign them respecting hierarchy
    assign_probs_topdown(dendrogram, probs)

    return dendrogram

print(make_rnd_dendrogram(8))