# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union
import os, re
import os.path as osp
import numpy as np

from torch import Tensor

from . import protein_constants as pc
from .protein import Protein, to_pdb


def aatype_to_seq(
        aatype: Union[np.ndarray, Tensor],
):
    return ''.join([pc.restypes_with_x[aa] for aa in aatype])

def ideal_atom_mask(
        prot: Protein,
) -> np.ndarray:
  """
  Computes an ideal atom mask.
  `Protein.atom_mask` typically is defined according to the atoms that are
  reported in the PDB. This function computes a mask according to heavy atoms
  that should be present in the given sequence of amino acids.
  Args:
    prot: `Protein` whose fields are `numpy.ndarray` objects.
  Returns:
    An ideal atom37 mask.
  """
  return pc.STANDARD_ATOM_MASK[prot.aatype]

def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype: np.ndarray,
        b_factors: Optional[np.ndarray] = None,
) -> Protein:
    assert (atom37.ndim == 3) and (atom37.shape[-2:] == (37, 3)), \
        f"Expected shape (N_res, 37. 3), but got {atom37.shape}."
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])

    return Protein(
        name = 'MDLP', # MDLDruglib Protein
        atom_positions = atom37,
        atom_mask = atom37_mask,
        aatype = aatype,
        residue_index = residue_index,
        chain_index = chain_index,
        b_factors = b_factors)

def _search_max_index(
        file_path,
) -> int:
    _dir = osp.dirname(file_path)
    _name = osp.basename(file_path)
    exists = [f for f in os.listdir(_dir) if _name in f]
    idxs = [0]
    for ex in exists:
        find = re.findall(r'_(\d+).pdb', ex)
        if len(find) > 0:
            idxs.append(int(find[0]))
    return max(idxs)

def write_prot_to_pdb(
        file_path: str,
        pos37_repr: np.ndarray,
        aatype: np.ndarray,
        b_factors: Optional[np.ndarray] = None,
        overwrite: bool = False,
        no_indexing: bool = False,
) -> str:
    """
    Write aatype and atom37 representation to pdb file
        (Support multiple protein positions recording).
    """
    save_path = file_path
    if not no_indexing:
        max_existing_idx = _search_max_index(file_path) if overwrite else 0
        save_path = file_path.replace('.pdb', f'_{max_existing_idx + 1}.pdb')

    with open(save_path, 'w') as f:
        if pos37_repr.ndim == 4:
            for t, pos37 in enumerate(pos37_repr):
                atom37_mask = np.sum(np.abs(pos37), axis = -1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask,
                    aatype = aatype, b_factors = b_factors)
                pdb_prot = to_pdb(prot, model = t + 1, add_end = False)
                f.write(pdb_prot)
        elif pos37_repr.ndim == 3:
            atom37_mask = np.sum(np.abs(pos37_repr), axis = -1) > 1e-7
            prot = create_full_prot(
                pos37_repr, atom37_mask,
                aatype=aatype, b_factors=b_factors)
            pdb_prot = to_pdb(prot, model = 1, add_end = False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {pos37_repr.shape}. '
                             f'(M, N, 37, 3) or (N, 37, 3) are allowed.')
        f.write('END\n')

    return save_path


