# Copyright (c) MDLDrugLib. All rights reserved.
import warnings
import os.path as osp
from typing import (
    Optional,  Sequence, List
)
import numpy as np
from easydict import EasyDict as ed

import torch
from torch import Tensor

from druglib.utils.obj import protein_constants as pc
from druglib.utils.torch_utils import np_repeat, tree_map, masked_mean

from .utils import build_torsion_edges
from ..builder import PIPELINES



@PIPELINES.register_module()
class PocketFinderDefault:
    """
    Pocket residue selection based on the ligand 3D position
        within the predefined cutoff (unit angstrom) or
        maximum neighbors number (Optional).
    Args:
        by_ligand3d: bool. Use ground truth ligand (from crystal ligand or docked ligand)
            This is highly suggested when training. High priority than args `point3d_obj`.
            Note that When by_ligand3d set to True, point3d_obj must be 'centroid' or 'all'.
        point3d_obj: Sequence, optional. Get pocket residue given the reference coordinates.
            A single 3d point list [x, y, z], coordinates file name
            in the protein file dir, file path, or specific string are supported,
            When input string is 'centroid' or 'all', args `by_ligand` must be True .
            When by_ligand3d set to False, it will first find whether there are
            pocket center float list or tuple input for pocket selection, if not,
            then find whether there are pocket ceneter file. Defaults to `centroid`.
            When set to None, we find the keyword `pocket_sel_center` as pocket selection
        ligand_coords: str. Get pocket residue given
            the reference coordinates. 'centroid' or 'all'.
            Defaults to 'centroid'.
        selection_mode: str. Pocket residue selection method.
            Defaults to 'any'. Only 'any', 'centroid' and `atom`
            mode will be supported. See `utils.obj.protein` for details.
        cutoff: int. Defaults to 10. The cutoff radius. Angstrom
            as the unit.
        max_neighbors: int, optional. The maximum number of residues selected.
            Defaults to None, indicating no upper bound.
    """

    def __init__(
            self,
            by_ligand3d: bool = True,  # high priority than point3d_obj
            point3d_obj: Optional[Sequence] = 'centroid',
            selection_mode: str = 'any',
            cutoff: int = 10,
            max_neighbors: Optional[int] = None,
    ):
        assert isinstance(point3d_obj, Sequence) or point3d_obj is None, \
            f'args `point3d_obj` must be type Sequence or None, but got {type(point3d_obj)}'
        if by_ligand3d and point3d_obj not in ['centroid', 'all']:
            raise ValueError(f"args `point3d_obj` ({point3d_obj}) must be 'centroid' or 'all' "
                             f"when args `by_ligand3d` ({by_ligand3d}) set to True.")

        self.by_ligand3d = by_ligand3d
        self.point3d_obj = point3d_obj
        self.selection_mode = selection_mode
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def parse_ligand(self, data):
        ligand = data['ligand']

        if self.point3d_obj == 'centroid':
            point3d = ligand.cm
            # Here we use the ligand centroid as randomized conformation center
            data['pocket_sel_center'] = torch.from_numpy(point3d).float()
        elif self.point3d_obj == 'all':
            point3d = ligand.atom_positions
            # Here we use the ligand 3d center as randomized conformation center
            data['pocket_sel_center'] = torch.from_numpy(point3d.mean(0)).float()
        else:
            raise ValueError(self.point3d_obj)

        assert data['pocket_sel_center'].shape == (3,), \
            f"Randomized conformation center shape should be (3, ), " \
            f"but got {data['pocket_sel_center'].shape}."

        return data, point3d

    def parse_centroid(self, data):
        point3d = self.point3d_obj

        if point3d is None:
            point3d = data.get('pocket_sel_center', None)
        elif point3d in ['centroid', 'all']:
            warnings.warn(f"when args `by_ligand3d` set to False, `point3d_obj` should not be 'centroid' or 'all'")
            if 'ligand' not in data:
                raise ValueError(f"`point3d_obj`set to {point3d}, ligand should be parsed early.")
            data, point3d = self.parse_ligand(data)
        elif isinstance(point3d, str):
            if not osp.exists(point3d):
                # when the input file name in the protein dir
                prot_dir = osp.dirname(data['protein_file'])
                point3d = osp.join(prot_dir, point3d)

            if osp.exists(point3d):
                with open(point3d, 'r') as f:
                    p3 = f.readlines()[0].strip().split(',')
                point3d = [float(_p3) for _p3 in p3[:3]]
            else:
                raise FileExistsError(f'Cannot find `point3d_obj` file from {point3d}')

        if isinstance(point3d, (list, tuple)):
            assert len(point3d) == 3, f'input list args `point3d_obj` ' \
                                      f'must be 3D point with format [x, y, z], but got {point3d}.'
            point3d = np.array(point3d, dtype=np.float)

        if point3d is None or not isinstance(point3d, (np.ndarray, Tensor)):
            raise ValueError(f'When args `point3d_obj` set to None, '
                             f'keyword `pocket_sel_center` (np.ndarray or Tensor) must be in data.')

        point3d = torch.FloatTensor(point3d)
        if not (len(point3d.shape) < 3 and point3d.shape[-1] == 3):
            raise ValueError("Only Shape (N, 3) or (3,) could be allowed for point3d to select pocket.")

        # parse pocket center position
        pocket_sel_center = point3d
        if len(pocket_sel_center.shape) == 2:
            pocket_sel_center = pocket_sel_center.mean(dim=0, keepdim=True)
        data['pocket_sel_center'] = pocket_sel_center

        return data, point3d

    def parse_point3d(self, data):
        if self.by_ligand3d:
            data, point3d = self.parse_ligand(data)
        else:
            data, point3d = self.parse_centroid(data)

        return data, point3d

    def process_pocket(self, data, pocket):
        data['pocket'] = pocket.to_dense()
        return data

    def __call__(self, data):
        protein = data['protein']
        data, point3d = self.parse_point3d(data)

        pocket, pocket_mask = protein.query_region(
            ref_coordinates=point3d,
            selection_mode=self.selection_mode,
            radius=self.cutoff,
            max_neighbors=self.max_neighbors,
            return_mask=True,
        )
        data['pocket_mask'] = torch.from_numpy(pocket_mask).bool()
        data = self.process_pocket(data, pocket)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'by_ligand3d={self.by_ligand3d}, '
                f'point3d_obj={self.point3d_obj}, '
                f'selection_mode={self.selection_mode}, '
                f'cutoff={self.cutoff}, '
                f'max_neighbors={self.max_neighbors})')

@PIPELINES.register_module()
class SCPocketFinderDefault(PocketFinderDefault):
    """Extract all atom representation of pocket"""
    def process_pocket(self, data, pocket):
        data['pocket'] = pocket  # for feature extraction
        template = pocket.extract_chi_and_template(return_radian = True)
        template = ed(tree_map(
            lambda x: torch.from_numpy(x).float(),
            template, np.ndarray))
        template.sequence = template.sequence.long()
        data.update(template)

        # consider missing atoms
        atom_positions, restype_atom14_mask = pocket.to_pos14(
            consider_missing_atoms = True)
        data['atom14_position'] = torch.from_numpy(atom_positions).float()
        data['atom14_mask'] = torch.from_numpy(restype_atom14_mask).bool().squeeze(-1)

        return data


@PIPELINES.register_module()
class PocketGraphBuilder:
    """
    Pocket side chain graph model construction.
    atom14 position repr and sequence vector are input.
    """
    def __call__(self, data):
        sequence = data['sequence']
        atom14_mask = data['atom14_mask']

        # torsion angle edge construction
        torsion_edge_index, chis_mask = build_torsion_edges(sequence, atom14_mask)
        torsion_edge_index = torsion_edge_index[..., 1, :]
        data['torsion_edge_index'] = torsion_edge_index
        data['sc_torsion_edge_mask'] = chis_mask

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}()')

@PIPELINES.register_module()
class PocketFeaturizer:
    """
    Support atom-level feature and residue (or Ca) level feature.
    Divide into residue-level Ca representation and atoms-level all-atoms representation.
    """
    def __init__(
            self,
            use_extra_feat: Optional[List[str]] = None,
    ):
        self.use_extra_feat = use_extra_feat
        self.eps = 1e-12

    def __call__(self, data):
        atom14_mask = data['atom14_mask']
        sequence = data['sequence']
        atoms37_to_atoms14 = pc.atoms37_to_atoms14_mapper[sequence]
        prot = data['pocket']
        num_res = prot.num_res()
        res_index = np.arange(num_res).reshape(-1, 1)

        def _atom37_to_atom14(arr: np.ndarray) -> Tensor:
            repeated_array = np_repeat(arr, num_res)
            repeated_array = repeated_array[res_index, atoms37_to_atoms14]
            return torch.from_numpy(repeated_array).float()

        # 1.1 all-atoms node feature (N_res, 14, ...)
        # 37 one-hot encoding
        atom37_label = _atom37_to_atom14(np.arange(prot.atom_mask.shape[-1]))
        # 22 one-hot encoding
        atomcoarse22_label = _atom37_to_atom14(pc.atom37_to_coarse_atom_type)
        # 4 one-hot encoding
        atom4_label = _atom37_to_atom14(pc.atom37_to_atom_element)
        # 21 one-hot encoding
        aatype21_label: Tensor = sequence.unsqueeze(-1).repeat(1, 14)
        # 2 one-hot encoding
        is_backbone = torch.zeros_like(aatype21_label, dtype = torch.float)
        is_backbone[:, :4] = 1.

        residue_prop = getattr(prot, 'residue_prop', None)
        if residue_prop is None:
            raise ValueError(f":cls:`{self.__class__.__name__}` currently needs the residue property in residue_prop")
        atoms_attr_list = [atom37_label, atomcoarse22_label, atom4_label, aatype21_label, is_backbone]

        # residue-depth suitable for rigid receptor docking
        for attr_nm in self.use_extra_feat or ['ss', 'phi', 'psi']:
            attr = residue_prop.get(attr_nm, None)
            if attr is not None:
                attr = torch.from_numpy(attr).view(-1, 1).float()
                if attr_nm in ['ss']:
                    atoms_attr_list.append(attr.repeat(1, 14))
        data['pocket_node_feature'] = torch.stack(
            atoms_attr_list, dim = -1
        ).float() * atom14_mask.unsqueeze(-1)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'use_extra_feat={self.use_extra_feat}, '
                f')')


@PIPELINES.register_module()
class Decentration:
    """Move the pocket (maybe with ligand) structure to pocket Ca center."""
    def __call__(self, data):
        atom14_pos = data['atom14_position']
        atom14_mask = data.get('atom14_mask', None)
        if atom14_mask is None:
            sequence: Tensor = data['sequence']
            restype_atom14_mask = pc.restype_atom14_mask[sequence]
            atom14_mask = torch.from_numpy(restype_atom14_mask).bool()
        atom14_mask = atom14_mask.unsqueeze(-1)

        pocket_center_pos = masked_mean(
            atom14_pos[:, 1],
            atom14_mask[:, 1],
            dim = 0, keepdim = True)
        data['pocket_center_pos'] = pocket_center_pos

        atom14_pos = atom14_pos - pocket_center_pos.view(1, 1, 3)
        data['atom14_position'] = atom14_pos * atom14_mask
        data['backbone_transl'] = data['backbone_transl'] - pocket_center_pos

        if 'lig_pos' in data:
            lig_pos = data['lig_pos']
            if lig_pos.ndim == 3:
                pocket_center_pos = pocket_center_pos.unsqueeze(0)
            elif lig_pos.ndim != 2:
                raise ValueError('lig_pos shape must be (M, N, 3) or (N, 3)')
            data['lig_pos'] = lig_pos - pocket_center_pos

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}()')