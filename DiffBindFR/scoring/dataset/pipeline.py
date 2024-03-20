# Copyright (c) MDLDrugLib. All rights reserved.
import os
import logging
from io import StringIO

import numpy as np
import pandas as pd

from rdkit import Chem

import prody
from prody import parsePDB, writePDBStream
# slience prody
prody.LOGGER._logger.setLevel(logging.INFO)

import torch
from torch_geometric.data import HeteroData

from .ligand_feature import get_ligand_feature
from .protein_feature import get_protein_feature


def generate_graph_4_Multi_PL_of(
        protein_pdb_path: str,
        ligand_mol: Chem.rdchem.Mol,
        cry_xyz: np.ndarray,
        pkt_radius: float = 12.0,
):
    # get pocket
    # cry_xyz = crystal_lig.GetConformer().GetPositions()
    pocket_center = torch.from_numpy(cry_xyz).to(torch.float32).mean(dim=0)
    protein_prody_obj = parsePDB(protein_pdb_path)
    condition = f'same residue as exwithin {pkt_radius} of somepoint'
    pocket_selected = protein_prody_obj.select(
        condition, somepoint=cry_xyz)
    output = StringIO()
    writePDBStream(output, pocket_selected)
    pkt_pdb_string = output.getvalue()

    l_xyz = torch.from_numpy(ligand_mol.GetConformer().GetPositions()).to(torch.float32)
    # only rescore pos
    rdkit_mol = ligand_mol
    # get feats
    (p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index,
     p_edge_s, p_edge_v) = get_protein_feature(pkt_pdb_string, pdb_string=True)

    (l_xyz_rdkit, l_node_feature, l_edge_index, l_edge_feature, l_full_edge_s,
     l_interaction_edge_mask, l_cov_edge_mask) = get_ligand_feature(rdkit_mol)
    # to data
    data = HeteroData()
    # protein
    data.pocket_center = pocket_center.view((1, 3)).to(torch.float32)
    data['protein'].node_s = p_node_s.to(torch.float32)
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32)
    data['protein'].xyz_full = p_xyz_full.to(torch.float32)
    data['protein'].seq = p_seq
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32)
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32)
    # ligand
    data['ligand'].xyz = l_xyz.to(torch.float32)
    data['ligand'].node_s = l_node_feature.to(torch.int32)
    data['ligand'].cov_edge_mask = l_cov_edge_mask
    data['ligand', 'l2l', 'ligand'].edge_index = l_edge_index.to(torch.long)
    data['ligand', 'l2l', 'ligand'].edge_s = l_edge_feature.to(torch.int32)
    data['ligand'].mol = rdkit_mol

    return data

class MolISNoneError(Exception):
    pass

def file2conformer(*args, sanitize = True):
    for f in args:
        try:
            if os.path.splitext(f)[-1] == '.sdf':
                mol = Chem.MolFromMolFile(f, removeHs=True, sanitize=sanitize)
            else:
                mol = Chem.MolFromMol2File(f, removeHs=True, sanitize=sanitize)
            if mol is not None:
                if not sanitize:
                    Chem.GetSSSR(mol)
                    Chem.GetSymmSSSR(mol)
                mol = Chem.RemoveAllHs(mol, sanitize=sanitize)
                return mol
            elif sanitize:
                mol = file2conformer(f, sanitize = False)
                if mol is not None:
                    return mol
        except:
            continue

def single_process(
        row: pd.Series,
        pocket_radius: float = 12.0
):
    cry_lig_sdf = row.get('crystal_ligand', None)
    pocket_center = row.get('center', None) # fmt: x,y,z
    if cry_lig_sdf is not None:
        cry_lig_mol2 = str(cry_lig_sdf).replace('.sdf', '.mol2')
        inp = (cry_lig_sdf, cry_lig_mol2) if cry_lig_mol2 is not None else (cry_lig_sdf,)
        cry_ligand_mol = file2conformer(*inp)
        cry_xyz = cry_ligand_mol.GetConformer().GetPositions()
    elif pocket_center is not None:
        cry_xyz = pocket_center.split(',')
        cry_xyz = [x.strip() for x in cry_xyz]
        cry_xyz = np.array([cry_xyz], dtype=np.float32)
        assert cry_xyz.ndim == 2, f'The dim of cry_xyz should be 2 but got {cry_xyz.ndim}'
    else:
        raise ValueError('Dataframe must have either crystal_ligand or center to define pocket.')

    protein_pdb = row['protein_pdb']
    ligand_sdf = row['docked_lig']
    docked_mol = file2conformer(ligand_sdf)

    try:
        hg = generate_graph_4_Multi_PL_of(
            protein_pdb,
            docked_mol,
            cry_xyz,
            pkt_radius=pocket_radius,
        )
    except Exception as e:
        print(protein_pdb, cry_lig_sdf, pocket_center, ligand_sdf, str(e))
        raise e

    return (hg, ) # make tuple as HeteroData (mapping object) does not fit dataframe output