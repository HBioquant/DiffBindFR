# Copyright (c) MDLDrugLib. All rights reserved.
import torch
from torch import Tensor
from rdkit import Chem
from .lrmsd import calc_rmsd
from ..utils import read_mol


def caltestset_rmsd(
        mol_pred_file,
        mol_true_file,
) -> float:
    mol_pred = read_mol(mol_pred_file)
    assert mol_pred is not None, mol_pred_file
    mol_pred = Chem.RemoveAllHs(mol_pred, sanitize=False)

    mol_true = read_mol(mol_true_file)
    assert mol_true is not None, mol_true_file
    mol_true = Chem.RemoveAllHs(mol_true, sanitize=False)

    rsd = calc_rmsd(mol_pred, mol_true)
    return rsd

def calc_lig_centroid(
        pred_pos: Tensor,
        target_pos: Tensor,
) -> float:
    pred_pos_mean = torch.mean(pred_pos, dim = -2)
    target_pos_mean = torch.mean(target_pos, dim = -2)
    dist = (pred_pos_mean - target_pos_mean).norm(dim = -1)
    return dist.item()

def caltestset_cdist(
        mol_pred_file,
        mol_true_file,
) -> float:
    mol_pred = read_mol(mol_pred_file)
    assert mol_pred is not None, mol_pred_file
    mol_pred = Chem.RemoveAllHs(mol_pred, sanitize=False)

    mol_true = read_mol(mol_true_file)
    assert mol_true is not None, mol_true_file
    mol_true = Chem.RemoveAllHs(mol_true, sanitize=False)

    cdist = calc_lig_centroid(
        torch.from_numpy(mol_pred.GetConformer(0).GetPositions()),
        torch.from_numpy(mol_true.GetConformer(0).GetPositions()),
    )
    return cdist

