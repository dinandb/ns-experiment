import igraph as ig
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.manifold import SpectralEmbedding
from superalgorithm import SuperAlgorithm

class Spectral(SuperAlgorithm):
    """Agglomerative community detection with spectral embedding."""

    def run(self, g: ig.Graph, linkage_method='average') -> list[tuple[int, int]]:
        components = g.connected_components()
        merges = []
        component_roots = []

        for component in components:
            component = list(component)
            if len(component) == 1:
                component_roots.append(component[0])
                continue

            subgraph = g.induced_subgraph(component)
            adj = np.asarray(subgraph.get_adjacency().data)
            n_embed = min(16, len(component) - 1)
            S = SpectralEmbedding(n_components=n_embed, affinity='precomputed').fit_transform(adj)
            D = linkage(S, linkage_method)

            id_map = {i: component[i] for i in range(len(component))}
            next_global = g.vcount() + len(merges)

            for step, (local_a, local_b, _, _) in enumerate(D):
                global_a = id_map[int(local_a)]
                global_b = id_map[int(local_b)]
                merges.append((global_a, global_b))
                id_map[len(component) + step] = next_global + step

            component_roots.append(next_global + len(D) - 1)

        while len(component_roots) > 1:
            a = component_roots.pop()
            b = component_roots.pop()
            merges.append((a, b))
            component_roots.append(g.vcount() + len(merges) - 1)

        return merges
