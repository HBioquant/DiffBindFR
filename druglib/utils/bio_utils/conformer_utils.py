# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union, List, Tuple
import warnings, copy
from easydict import EasyDict as ed
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as R
import torch
from torch import Tensor, LongTensor
from torch_scatter import scatter_add

from rdkit import Chem
from rdkit.Chem import AllChem

from ..geometry_utils.utils import axis_angle_to_rot, radian2sincos
from ..geometry_utils.superimposition import rigid_transform_Kabsch_3D_torch
from ..torch_utils import (
    slice_tensor_batch, ptr_and_batch,
)
from ..obj.prot_math import build_pdb_from_template



def generate_multiple_conformers(
        mol: Chem.rdchem.Mol,
        max_num_conf: int = 2,
        num_pool: int = 10,
        force_field: str = 'MMFF94',
        rmsd_thr: float = 0.5,
) -> Chem.rdchem.Mol:
    """
    Generate multiple conformers for molecules.
    Predefine the number of conformers pool,  we calculate
        the energy and rmsd to select the best molecule
        conformers for the diversity.
    Args:
        mol: Chem.rdchem.Mol.
        max_num_conf: int. The max number of generated conformers,
            which will be selected from the conformers pool.
            Defaults to 2.
        num_pool: int. The number of conformers in the pool.
            Defaults to 10.
        force_field: str. The force field to optimize the molecule
            conformers and calculate the energies.
            Choices are 'MMFF94', 'MMFF94s' and 'UFF'.
            Defaults to 'MMFF94'
        rmsd_thr: float. The threshold of rmsd between the higher
            energy conformer and the lower energy conformers.
            Defaults to 0.5.
            Lower value may result in not enough conformations being
            generated.
    Returns:
        Chem.rdchem.Mol with conformers.
    """
    force_field = force_field.upper()
    if force_field not in ['MMFF94', 'MMFF94s', 'UFF']:
        raise ValueError('Currently, MMFF94, MMFF94s and UFF force field are supported.')
    if not isinstance(rmsd_thr, (int, float)) or rmsd_thr > 1:
        raise ValueError('Args `rmsd_thr` should be int or float type and <= 1.')
    if max_num_conf > num_pool:
        raise ValueError('Args `max_num_conf` should be smaller than args `num_pool`.')

    mol = Chem.AddHs(mol)
    schme = AllChem.ETKDGv2()
    ids = AllChem.EmbedMultipleConfs(mol, num_pool, schme)
    if -1 in ids:
        schme.useRandomCoords = True
        AllChem.EmbedMultipleConfs(mol, num_pool, schme)

    if not mol.GetNumConformers():
        raise RuntimeError("Generate multiple molecules is failed.")
    if force_field.startswith('MMFF'):
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant=force_field)
    elif force_field == 'UFF':
        AllChem.UFFOptimizeMoleculeConfs(mol)

    if mol.GetNumConformers() < max_num_conf:
        warnings.warn(f'Failed to get {max_num_conf} molecules conformers, only {mol.GetNumConformers()} available')
        return mol

    energies = []
    for conf in mol.GetConformers():
        if force_field == 'UFF':
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        elif force_field.startswith('MMFF'):
            prop = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=force_field)
            ff = AllChem.MMFFGetMoleculeForceField(mol, pyMMFFMolProperties=prop, confId=conf.GetId())
        energies.append(ff.CalcEnergy())
    energies = np.array(energies, dtype = np.float32)
    ascending = np.argsort(energies)

    N = mol.GetNumConformers()
    rmsd = np.zeros(shape = (N, N), dtype = np.float32)
    for i, ref in enumerate(mol.GetConformers()):
        for j, fit in enumerate(mol.GetConformers()):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref.GetId(), fit.GetId())
            rmsd[j, i] = rmsd[i, j]

    keep = []
    for i in ascending:
        if len(keep) == 0:
            keep.append(i)
        if len(keep) > max_num_conf:
            continue
        _rmsd = rmsd[i][np.array(keep, dtype = int)]
        if (_rmsd >= rmsd_thr).all():
            keep.append(i)

    new_one = Chem.Mol(mol)
    new_one.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for i in keep:
        conf = mol.GetConformer(conf_ids[i])
        new_one.AddConformer(conf, assignId = True)
    return new_one

def conformer_generation(
        mol: Chem.rdchem.Mol,
        num_pool: int = 10,
        force_field: str = 'MMFF94',
) -> Chem.rdchem.Mol:
    """
    Generate a low-energy conformer from the pool,
        so the obtained molecule is at the low energt
        state as possible.
    Compared to the :function:`fast_conformer_generation`,
        this is more exhaustive search for lower energy conformers.
    Args:
        mol: Chem.rdchem.Mol.
        num_pool: int. The number of conformers in the pool.
            Defaults to 10.
        force_field: str. The force field to optimize the molecule
            conformers and calculate the energies.
            Choices are 'MMFF94', 'MMFF94s' and 'UFF'.
            Defaults to 'MMFF94'
    Returns:
        Chem.rdchem.Mol with conformer.
    """
    return generate_multiple_conformers(
        mol = mol,
        max_num_conf = 1,
        num_pool = num_pool,
        force_field = force_field,
        rmsd_thr = 0,
    )

def fast_conformer_generation(
        mol: Chem.rdchem.Mol,
        force_field: str = 'MMFF94s',
        seed: Optional[int] = None,
) -> Chem.rdchem.Mol:
    """
    Fast conformer generation.
    This is neccessary for coarse-grained virtual screening.
    Args:
        mol: Chem.rdchem.Mol.
        force_field: str. The force field to optimize the molecule
            conformers and calculate the energies.
            Choices are 'MMFF94', 'MMFF94s' and 'UFF'.
            Defaults to 'MMFF94'
        Seed: int, Optional. Defaults to None.
    Returns:
        Chem.rdchem.Mol with conformer.
    """
    force_field = force_field.upper()
    if force_field not in ['MMFF94', 'MMFF94s', 'UFF']:
        raise ValueError('Currently, MMFF94, MMFF94s and UFF force field are supported.')
    mol = simple_conformer_generation(mol, seed)

    if force_field.startswith('MMFF'):
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant = force_field)
    elif force_field == 'UFF':
        AllChem.UFFOptimizeMolecule(mol)

    return mol

def simple_conformer_generation(
        mol: Chem.rdchem.Mol,
        seed: Optional[int] = None,
) -> Chem.rdchem.Mol:
    """
    Simple conformer generation using ETKDGv2 to Embed the mol object to 3D
    Args:
        mol: Chem.rdchem.Mol.
        Seed: int, Optional. Defaults to None.
    Returns:
        Chem.rdchem.Mol with conformer.
    """
    mol = Chem.AddHs(mol)
    schme = AllChem.ETKDGv2()
    if seed is not None:
        schme.randomSeed = seed
    id = AllChem.EmbedMolecule(mol, schme)

    if id == -1:
        # use random coords try again
        schme.useRandomCoords = True
        schme.maxIterations = 5000
        AllChem.EmbedMolecule(mol, schme)

    return mol

class ConformerGenerationError(Exception):
    pass

def fast_generate_conformers_onebyone(
        mol: Chem.rdchem.Mol,
        num_confs: int = 10,
        force_field: str = 'MMFF94',
        tolerance: int = 200,
) -> Chem.rdchem.Mol:
    """
    The above three functions cannot guarantee
        that the target number of conformations
        can be accurately generated.
    This function is used to quickly generate the
        target number of conformations one by one.
    Args:
        mol: Chem.rdchem.Mol.
        num_confs: int. The exact number of conformers.
            Defaults to 10.
        force_field: str. The force field to optimize the molecule
            conformers and calculate the energies.
            Choices are 'MMFF94', 'MMFF94s' and 'UFF'.
            Defaults to 'MMFF94'
        tolerance: int. The exhast max number. Defaults to 200.
    Returns:
        Chem.rdchem.Mol with conformers.
    """
    conformers = []
    exhaust = 0
    new_one = Chem.Mol(mol)
    new_one.RemoveAllConformers()
    while len(conformers) < num_confs:
        try:
            mol = fast_conformer_generation(
                copy.deepcopy(mol),
                force_field = force_field
            )
            conformers.append(mol.GetConformer())
        except ConformerGenerationError as e:
            if exhaust == 0:
                warnings.warn(str(e))
            if exhaust > tolerance:
                raise ConformerGenerationError(str(e))
            exhaust += 1

    for conf in conformers:
        new_one.AddConformer(conf, assignId = True)

    return new_one

def get_ligconf(
        mol:Chem.rdchem.Mol,
        allow_genconf: bool = True,
) -> Optional[np.ndarray]:
    NC = mol.GetNumConformers()
    if NC < 0.5:
        if not allow_genconf:
            return None
        mol = fast_conformer_generation(mol, force_field = 'MMFF94s')
    atom_positions = mol.GetConformer(0).GetPositions()

    return atom_positions

def remove_all_hs(
        mol:Chem.rdchem.Mol,
) -> Chem.rdchem.Mol:
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return Chem.RemoveHs(mol, params)

def get_pos_from_mol(
        mol: Chem.rdchem.Mol,
) -> ndarray:
    """
    Extract the numpy array of coordinates from a molecule :obj:`Chem.rdchem.Mol`.
    Args:
        mol: Chem.rdchem.Mol. rdkit molecule.
    Returns: the numpy coordinates of the input molecule.
    """
    pos = np.zeros((mol.GetNumAtoms(), 3)).astype('double')
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        pos[i, 0] = position.x
        pos[i, 1] = position.y
        pos[i, 2] = position.z
    return pos

def modify_conformer_torsion_angles(
        pos: Tensor, # (N_atoms, 3)
        edge_index: Tensor, # (N_rot_bond, 2)
        rot_node_mask: Tensor, # (N_rot_bond, N_atoms)
        torsion_updates: Union[Tensor, ndarray], # (N_rot_bond, )
) -> Tensor:
    pos = pos.clone().detach()

    for idx_edge, e in enumerate(edge_index):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not rot_node_mask[idx_edge, u]
        assert rot_node_mask[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / torch.linalg.norm(rot_vec)
        rot_mat = axis_angle_to_rot(rot_vec)

        pos[rot_node_mask[idx_edge]] = (pos[rot_node_mask[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    return pos

def modify_conformer(
        pos: Tensor,
        edge_index: Tensor,
        tor_edge_mask: Tensor,
        rot_node_mask: Tensor,
        tr_update: Tensor,
        rot_update: Tensor,
        torsion_updates: Optional[Tensor] = None,
) -> Tensor:
    lig_center = pos.mean(dim = 0, keepdim = True)
    rot_mat = axis_angle_to_rot(rot_update.squeeze())
    rigid_new_pos = (pos - lig_center) @ rot_mat.T + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(
            rigid_new_pos,
            edge_index.T[tor_edge_mask],
            rot_node_mask,
            torsion_updates).to(rigid_new_pos.device)
        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        pos = aligned_flexible_pos
    else:
        pos = rigid_new_pos

    return pos

def randomize_lig_pos(
        pos: Tensor,
        edge_index: Tensor,
        tor_edge_mask: Tensor,
        rot_node_mask: Tensor,
        tr_max: int = 10,
        no_tr: bool = False,
) -> Tensor:
    """
    Randomly sample rigid ligand centroid translation \delta X ~ N(),
        center-free rotation update
    """
    num_torsion_bonds = tor_edge_mask.sum()
    if not isinstance(rot_node_mask, torch.BoolTensor):
        rot_node_mask = rot_node_mask.bool()
    tor_edge_mask = tor_edge_mask.bool()
    if num_torsion_bonds > 0:
        torsion_updates = np.random.uniform(
            low = -np.pi, high = np.pi, size = num_torsion_bonds)
        pos = modify_conformer_torsion_angles(
            pos, edge_index.T[tor_edge_mask],
            rot_node_mask, torsion_updates)

    lig_center = pos.mean(dim = 0, keepdim = True)
    rot_update = torch.from_numpy(R.random().as_matrix()).float()
    pos = (pos - lig_center) @ rot_update.T

    if not no_tr:
        tr_update = torch.normal(mean = 0, std = tr_max, size = (1, 3))
        pos = pos + tr_update

    return pos

def randomize_batchlig_pos(
        pos: Tensor,
        edge_index: Tensor,
        tor_edge_mask: Tensor,
        rot_node_mask: List[Tensor],
        batch: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        tr_max: int = 10,
        no_tr: bool = False,
) -> Tensor:
    batch, ptr = ptr_and_batch(batch, ptr)
    pos_list = slice_tensor_batch(pos, batch)
    edge_batch = batch[edge_index[0]]
    tor_edge_mask_list = slice_tensor_batch(tor_edge_mask, edge_batch)
    idx0 = edge_index[0] - ptr[batch][edge_index[0]]
    idx1 = edge_index[1] - ptr[batch][edge_index[1]]
    edge_index_single = torch.stack([idx0, idx1], dim = 1)
    edge_index_list = slice_tensor_batch(edge_index_single, edge_batch)
    new_pos_list = []
    for _pos, _edge_index, _tor_edge_mask, _rot_node_mask in zip(
        pos_list, edge_index_list, tor_edge_mask_list, rot_node_mask
    ):
        new_pos = randomize_lig_pos(
            _pos, _edge_index.T, _tor_edge_mask,
            _rot_node_mask, tr_max = tr_max, no_tr = no_tr)
        new_pos_list.append(new_pos)

    return torch.cat(new_pos_list, dim = 0)


def update_batchlig_pos(
        tr_update: Tensor,
        rot_update: Tensor,
        torsion_updates: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        tor_edge_mask: Tensor,
        rot_node_mask: List[Tensor],
        batch: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
) -> Tensor:
    batch, ptr = ptr_and_batch(batch, ptr)
    batch_size = batch.max().item() + 1
    assert tr_update.shape[0] == batch_size
    assert len(rot_node_mask) == batch_size
    num_tor_bonds = tor_edge_mask.sum()
    assert torsion_updates.shape[0] == num_tor_bonds

    pos_list = slice_tensor_batch(pos, batch)
    edge_batch = batch[edge_index[0]]
    tor_edge_mask_list = slice_tensor_batch(tor_edge_mask, edge_batch)
    tor_bonds_batch = batch[edge_index[0]][tor_edge_mask.bool()]
    num_tor_bonds = scatter_add(
        batch.new_ones(tor_bonds_batch.shape).long(),
        tor_bonds_batch, dim = 0, dim_size = batch_size,
    )
    num_torb_cumsum = num_tor_bonds.cumsum(-1)
    num_torb_cumsum = torch.cat([num_torb_cumsum.new_zeros((1,)), num_torb_cumsum], dim = 0)
    idx0 = edge_index[0] - ptr[batch][edge_index[0]]
    idx1 = edge_index[1] - ptr[batch][edge_index[1]]
    edge_index_single = torch.stack([idx0, idx1], dim = 1)
    edge_index_list = slice_tensor_batch(edge_index_single, edge_batch)
    new_pos_list = []
    for idx, (_pos, _edge_index, _tor_edge_mask, _rot_node_mask) in enumerate(zip(
            pos_list, edge_index_list, tor_edge_mask_list, rot_node_mask
    )):
        _tr_update = tr_update[idx:idx + 1]
        _rot_update = rot_update[idx]
        _num_tor_bonds = num_tor_bonds[idx]
        _tor_update = None
        if _num_tor_bonds > 0:
            _tor_update = torsion_updates[num_torb_cumsum[idx]:num_torb_cumsum[idx + 1]]
        new_pos = modify_conformer(
            _pos,
            _edge_index.T,
            _tor_edge_mask.bool(),
            _rot_node_mask.bool(),
            _tr_update,
            _rot_update,
            torsion_updates = _tor_update,
        )
        new_pos_list.append(new_pos)

    return torch.cat(new_pos_list, dim=0)

def randomize_sc_dihedral(
        sequence: Union[LongTensor, ndarray],
        backbone_transl: Union[Tensor, ndarray],
        backbone_rots: Union[Tensor, ndarray],
        default_frame: Union[Tensor, ndarray],
        rigid_group_positions: Union[Tensor, ndarray],
        psi_radian: Union[Tensor, ndarray],
        device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Randomly sample side chain at most 4 chis from U(-pi, pi) \in R
        and build all-atoms representation pos14 for the protein model.
    """
    if (device is None) and isinstance(sequence, Tensor):
        device = sequence.device()
    if device is None:
        device = torch.device('cpu')
    if isinstance(psi_radian, Tensor):
        psi_radian = psi_radian.cpu().numpy()
    if len(psi_radian.shape) == 1:
        psi_radian = psi_radian.reshape(-1, 1)
    assert len(psi_radian.shape) == 2, 'psi_radian shape must be (N, 1).'

    chis_update = np.random.uniform(
        low = -np.pi, high = np.pi,
        size = (sequence.shape[0], 4))
    chis_update = np.concatenate([chis_update, psi_radian], axis = 1)

    pos14, mask14 = build_pdb_from_template(
        ed(sequence = sequence,
           backbone_transl = backbone_transl,
           backbone_rots = backbone_rots,
           default_frame = default_frame,
           rigid_group_positions = rigid_group_positions,
           torsion_angle = radian2sincos(chis_update),
           ), device)

    return pos14 * mask14, mask14

