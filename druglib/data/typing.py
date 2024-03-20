# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional,
    Dict,
    Union,
    List,
    Tuple,
)
from enum import Enum

import torch
from torch import Tensor

from torch_sparse import SparseTensor
from torch_geometric.typing import (
    # Node-types are denoted by a single string, e.g.: `data['paper']`:
    NodeType,
    # Edge-types are denotes by a triplet of strings, e.g.:
    # `data[('author', 'writes', 'paper')]
    EdgeType,
    # There exist some short-cuts to query edge-types (given that the full triplet
    # can be uniquely reconstructed, e.g.:
    # * via str: `data['writes']`
    # * via Tuple[str, str]: `data[('author', 'paper')]`
    QueryType,
    Metadata,
    # Types for message passing
    Adj,
    OptTensor,
    PairTensor,
    OptPairTensor,
    PairOptTensor,
    Size,
    NoneType,
)
from .data_container import DataContainer, IndexType

# TODO note that "y" maybe graph-level label
# absolute words for the data type assignment
NODEWORLD = ["x", "pos", "y", "node_attr", "batch", "node_feature", "ptr"]
EDGEWORLD = ["edge_index", "edge_weight", "edge_attr", "edge_feature",
             "adj", "adj_t", "face", "coo", "csc", "csr"]
CVWORLD = ["img", "bbox", "cvmask", "cvlabel", "proposal", "seg"]

# key words to search the other data, so-called `relative` words
NODEKEYS = ["node", "pos"]
EDGEKEYS = ["edge", "adj", "face"]
CVKEYS = ["img", "bbox", "cvmask", "cvlabel", "proposal", "seg", "imgcls"]

class DataType(Enum):
    CV = 'cv'
    NODE = 'node'
    EDGE = 'edge'
    GRAPH = "graph"
    META = "meta"

class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'

# typing that is missed in importing PyTorch Geometric but has released in the github
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/typing.py
# A representation of a feature tensor
FeatureTensorType = Union[Tensor, SparseTensor, DataContainer]

# A representation of an edge index, following the possible formats:
#   * COO: (row, col)
#   * CSC: (row, colptr)
#   * CSR: (rowptr, col)
EdgeTensorType = Tuple[Tensor, Tensor]

# Types for sampling
InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]

# CV types
CVType = str

# Data input datatype
OptInput = Union[None, Tensor, DataContainer, SparseTensor]