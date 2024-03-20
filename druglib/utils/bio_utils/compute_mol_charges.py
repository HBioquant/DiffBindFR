# Copyright (c) MDLDrugLib. All rights reserved.
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from ..logger import print_log


def compute_mol_charges(
        mol: Chem.rdchem.Mol,
) -> None:
    """
    Compute molecule :obj:`Chem.rdchem.Mol` Gasteiger charges
    Args:
        rdkit molecule.
    """
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception as e:
        print_log("Unable to compute Gasteiger charges.")
        raise RuntimeError(e)

def get_atom_partial_charge(
        atom: Chem.Atom,
) -> float:
    """
    Get atom :obj:`Chem.Atom` Gasteiger charges
    Args:
        rdkit atom.
    E.g.
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCCCCC")
        >>> atom = mol.GetAtoms()[0]
        >>> get_atom_partial_charge(atom)
    """
    if isinstance(atom, Chem.Atom):
        try:
            value = atom.GetDoubleProp(str("_GasteigerCharge"))
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        except KeyError:
            return 0.0
    else:
        raise TypeError(f"Input must be rdkit.Chem.Atom, but got {type(atom)}")

