# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional,List,
    Tuple, Sequence,
    Mapping,
)
import warnings
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import numpy as np

import torch
from torch import Tensor


def tree_compare(
        x, y,
) -> bool:
    assert type(x) == type(y), \
        f'x {type(x)} and y {type(y)} type mismatch'
    if isinstance(x, (np.ndarray, Tensor)):
        return (x == y).all()
    elif isinstance(x, Sequence) and not isinstance(x, str):
        return all([tree_compare(_x, _y) for _x, _y in zip(x, y)])
    elif isinstance(x, Mapping):
        assert len(x) == len(y), f'Mapping type x ({len(x)}) and y ({len(y)}) length mismatch'
        return all([tree_compare(x[k], y[k]) for k in x.keys()])
    else:
        return x == y


def node_match_aprops(
        node1,
        node2,
) -> bool:
    """
    Check if atomic properties for two nodes match.
    """
    node1_props = node1["aprops"]
    node2_props = node2["aprops"]
    return tree_compare(node1_props, node2_props)

def edge_match_aprops(
        edge1,
        edge2,
) -> bool:
    """Check if edge properties for two edges match"""
    edge1_props = edge1["eprops"]
    edge2_props = edge2["eprops"]
    return tree_compare(edge1_props, edge2_props)

def to_nx(
        node_attrs: np.ndarray,
        edge_index: np.ndarray,
        edge_attrs: Optional[np.ndarray] = None,
) -> nx.Graph:
    num_nodes = node_attrs.shape[0]
    G = nx.Graph()

    for atid in range(num_nodes):
        G.add_node(
            atid,
            aprops = node_attrs[atid],
        )

    for i, (u, v) in enumerate(edge_index.T.tolist()):
        if u > v:
            continue
        if u == v:
            continue
        edge_attr = {}
        if edge_attrs is not None:
            edge_attr['eprops'] = edge_attrs[i]
        G.add_edge(u, v, **edge_attr)

    if not nx.is_connected(G):
        warnings.warn('Disconnected graph detected.')

    return G

def match_graphs(
        G1: nx.Graph,
        G2: nx.Graph,
        keep_self: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if (
            nx.get_node_attributes(G1, "aprops") == {}
            or nx.get_node_attributes(G2, "aprops") == {}
    ):
        # Nodes without atomic number information
        # No node-matching check
        node_match = None
        warnings.warn("No atomic property information stored on nodes. "
                      "Node matching is not performed...")

    else:
        node_match = node_match_aprops

    if (
            nx.get_edge_attributes(G1, "eprops") == {}
            or nx.get_edge_attributes(G2, "eprops") == {}
    ):
        edge_match = None
    else:
        edge_match = edge_match_aprops

    GM = GraphMatcher(G1, G2, node_match, edge_match)

    # Check if graphs are actually isomorphic
    if not GM.is_isomorphic():
        raise ValueError("Graphs are not isomorphic.\nMake sure graphs have the same connectivity.")

    def sorted_array(
            seq1: List[int],
            seq2: List[int],
    ):
        """Sort seq1 by seq2 and deprecate seq1 because always \equiv np.arange(num_nodes)"""
        assert len(seq1) == len(seq2), f'seq1 ({len(seq1)}) and seq2 ({len(seq2)}) length mismatch'
        seq1 = np.array(seq1)
        seq2 = np.array(seq2)
        sort_ind = np.argsort(seq2)
        seq1 = seq1[sort_ind]
        seq2 = seq2[sort_ind]
        assert (seq2 == np.arange(seq2.shape[0])).all(), \
            f'Graph matching node missing {seq2} while {np.arange(seq2.shape[0])} are needed'
        return seq1, seq2

    isomorphisms = []
    for isomorphism in GM.isomorphisms_iter():
        isom_arr, arange_arr = sorted_array(
            list(isomorphism.keys()), list(isomorphism.values()),
        )
        self = (isom_arr == arange_arr).all()
        if not self or keep_self:
            isomorphisms.append((isom_arr, arange_arr))

    return isomorphisms

