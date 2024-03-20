# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, List
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

from druglib.utils.obj import ligand_constants as lc
from druglib.utils.obj import Ligand3D

from .utils import find_torsion
from ..builder import PIPELINES


@PIPELINES.register_module()
class LigandFeaturizer:
    """
    Ligand graph node and edge featurizer.
    Use the user-specified keys as the feature.
    Args:
        node_feature_keys: a list of string (optional), which
            should be stored in the :obj:`Ligand3D` :attr:`atom_prop`.
        edge_feature_keys: a list of string (optional), which
            should be stored in the :obj:`Ligand3D` :attr:`bond_prop`.
        If any keys are not found, raise ValueError.
        If keeping None, all keys will be used.
    Save 'lig_node' (Tensor, unique) in the dict data.
    Save 'lig_edge_feat' (Tensor, unique) in the dict data.
        bond prop: bondstereo (scalar/6), isinring (T/F), isconjugated (T/F),
            bond_label (scalar, 0: covalent edge, 1: ring edge, 2: two-hop edge,
                3: knn edge)
    """
    def __init__(
            self,
            node_feature_keys: Optional[List[str]] = None,
            edge_feature_keys: Optional[List[str]] = None,
    ):
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys

    def __call__(self, data):
        Ligand: Ligand3D = data['ligand']

        # 1. atom feature (24 / 25 with NumHs)
        atom_prop = Ligand.atom_prop
        ALL_KEYS = list(atom_prop.keys())
        node_feature_list = list()
        for k in (self.node_feature_keys or ALL_KEYS):
            if k not in ALL_KEYS:
                raise ValueError(f'The specified keys {k} not in the `atom_prop` keys')
            v = atom_prop[k]
            if len(v.shape) == 1:
                node_feature_list.append(torch.from_numpy(v.reshape(-1, 1)).float())
                continue
            node_feature_list.append(torch.from_numpy(v).float())
        concat_node_feature: Tensor = torch.cat(
            node_feature_list, dim = -1)
        data['lig_node'] = concat_node_feature

        # 2. edge feature
        bond_prop = Ligand.bond_prop
        ALL_KEYS = list(bond_prop.keys())
        edge_feature_list = list()
        # bond type is necessary, using one-hot encoding
        bond_type = torch.from_numpy(Ligand.bond_type).long()
        bond_type = F.one_hot(bond_type, num_classes = lc.num_connecttypes) # 6 classes
        edge_feature_list.append(bond_type)
        for k in (self.edge_feature_keys or ALL_KEYS):
            if k not in ALL_KEYS:
                raise ValueError(f'The specified keys {k} not in the `atom_prop` keys')
            v = bond_prop[k]
            if len(v.shape) == 1:
                edge_feature_list.append(torch.tensor(v.reshape(-1, 1), dtype = torch.float32))
                continue
            edge_feature_list.append(torch.tensor(v, dtype = torch.float32))
        concat_edge_feature: Tensor = torch.cat(
            edge_feature_list, dim = -1)
        data['lig_edge_feat'] = concat_edge_feature

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'node_feature_keys={self.node_feature_keys}, '
                f'edge_feature_keys={self.edge_feature_keys}'
                f')')


@PIPELINES.register_module()
class TorsionFactory:
    """
    Find torsion bonds from molecule graph,
        the torsion bonds was defined as
        `lead-to-disconnected edges`.
    The algorithm is extended to `NoneType` egde
        so the class can safely connected with the
        :cls:`LigandGraphBuilder` and other extra edge.
    Because the torsion bonds should be found in the covalent
        graph, so make sure there are marks to identify the
        covalent edges.
    Make sure the ligand are compound without salt or coordinated metal.
    """
    def __call__(self, data):
        ligand = data['ligand']
        tor_edge_mask, rot_node_mask = find_torsion(ligand)
        data['tor_edge_mask'] = tor_edge_mask
        data['rot_node_mask'] = rot_node_mask

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}()')


@PIPELINES.register_module()
class LigandGrapher:
    """Get ligand topological graph from :object:`Ligand3D` with torsion."""
    def __call__(self, data):
        ligand: Ligand3D = data['ligand']
        data['lig_pos'] = torch.from_numpy(ligand.atom_positions).float() # (num_ligatoms, 3)
        data['lig_edge_index'] = torch.from_numpy(ligand.edge_index).long() # (2, num_edges_lig)
        data['tor_edge_mask'] = torch.from_numpy(data['tor_edge_mask']).long() # （num_edges_lig, )

        data['metastore'] = dict(
            num_nodes = data['lig_node'].size(0),
            num_edges = data['lig_edge_index'].size(-1))
        rot_node_mask: np.ndarray = data.pop('rot_node_mask')  # (unknown, num_ligatoms)
        data['metastore']['rot_node_mask'] = rot_node_mask.astype('bool')

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}()')