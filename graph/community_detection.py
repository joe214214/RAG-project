from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import igraph as ig
import leidenalg
import networkx as nx


@dataclass
class CommunityResult:
    node_to_community: Dict[str, int]
    num_communities: int


def networkx_to_igraph(graph: nx.Graph) -> ig.Graph:
    g = ig.Graph()
    g.add_vertices(list(graph.nodes))
    edges = [(u, v) for u, v in graph.edges]
    g.add_edges(edges)
    if graph.number_of_edges() > 0:
        weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges]
        g.es["weight"] = weights
    return g


def run_leiden(graph: nx.Graph, resolution: float = 1.0) -> CommunityResult:
    """
    Run Leiden community detection and return a mapping from node -> community id.
    """
    ig_graph = networkx_to_igraph(graph)
    # In igraph, edge attributes are accessed via item syntax, not `.get`.
    weights = ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.CPMVertexPartition,
        resolution_parameter=resolution,
        weights=weights,
    )
    node_to_community = {
        node: community_id for node, community_id in zip(ig_graph.vs["name"], partition.membership)
    }
    return CommunityResult(node_to_community=node_to_community, num_communities=len(partition))
