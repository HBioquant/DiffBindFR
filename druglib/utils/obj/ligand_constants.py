# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    List, Optional, Union,
    Tuple, Dict,
)
import re
import os.path as osp
from importlib import resources
from collections import defaultdict
import numpy as np
import functools
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, GetPeriodicTable, RDConfig
pt = GetPeriodicTable()

###################################### atom property
# the frequent ligand atoms, other atoms (drug-less) will be categorized to `other`
# consider implicit hydrogen
atom_types = [
    'C',
    'N',
    'O',
    'S',
    'F',
    'Cl',
    'Br',
    'I',
    'P',
    'Si',
    'B',
    'other',
]
num_atoms = len(atom_types) # := 12
atomtypes_to_id = {v: k for k, v in enumerate(atom_types)}

atomid_types = [pt.GetAtomicNumber(atom.capitalize()) for atom in atom_types[:-1]] + [-1]
num_atomids = num_atoms
atomidx_to_id = {v: k for k, v in enumerate(atomid_types)}

# consider explicit hydrogen
atom_types_with_H = atom_types + ['H'] # := 13
num_atoms_with_H = len(atom_types_with_H)
atomtypes_with_H_to_id = {v: k for k, v in enumerate(atom_types_with_H)}

# the atom hybridization types
hybridization_types = [
    'SP',
    'SP2',
    'SP3',
    'SP3D',
    'SP3D2',
    'other',
]
num_hybridizations = len(hybridization_types)
hybridization_to_id = {v: k for k, v in enumerate(hybridization_types)}

# the atom total degree
degree_types = list(range(10)) + ['other']
num_degrees = len(degree_types)
degree_to_id = {v: k for k, v in enumerate(degree_types)}

# the atom formal charge
fc_types = list(range(-5, 5)) + ['other']
num_fcs = len(fc_types)
fc_to_id = {v: k for k, v in enumerate(fc_types)}

# the number of radical electron
num_radical_electrons = list(range(5)) + ['other']
num_re = len(num_radical_electrons)
nre_to_id = {v: k for k, v in enumerate(num_radical_electrons)}

# include total num hydrogen types
total_num_hydrogen = list(range(9)) + ['other']
num_total_num_hydrogen = len(total_num_hydrogen)
total_num_hydrogen_to_id = {v: k for k, v in enumerate(total_num_hydrogen)}

# the chiral center types
chiral_types = [
    'R',
    'S',
    'other',
]
num_chirals = len(chiral_types)
chiral_types_to_id = {v: k for k, v in enumerate(chiral_types)}

# the CHI chiral types
CHI_chiral_types = [
    'CHI_UNSPECIFIED',
    'CHI_TETRAHEDRAL_CW',
    'CHI_TETRAHEDRAL_CCW',
    'CHI_OTHER',
    'CHI_TETRAHEDRAL',
    'CHI_ALLENE',
    'CHI_SQUAREPLANAR',
    'CHI_TRIGONALBIPYRAMIDAL',
    'CHI_OCTAHEDRAL',
]
num_chi_chirals = len(CHI_chiral_types)
chi_chiral_types_to_id = {v: k for k, v in enumerate(CHI_chiral_types)}

# the total number of hydrogen connected to the atom
numh_types = list(range(5))
num_numhs = len(numh_types)

# the possible number of rings in the ligand
numring_types = list(range(7)) + ['other']
num_numrings = len(numring_types)

# atom ring env
def IsInRingofN(
        ring: Chem.RingInfo,
        atomid: int,
) -> List[int]:
    return [
        int(ring.IsAtomInRingOfSize(atomid, N)) for N in ringsize_types
    ]

# the possible ring size (for display use only)
ringsize_types = list(range(3, 9))
num_ringsize = len(ringsize_types)

# the atom feature family
fdefName = osp.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
Factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
# iteratively get atom's feature: factory.GetFeaturesForMol(mol): iter(feature)
# feature.GetFamily()
# Atom Feature Family
aff_types = [
    'Acceptor',
    'Donor',
    'Aromatic',
    'Hydrophobe',
    'LumpedHydrophobe',
    'NegIonizable',
    'PosIonizable',
    'ZnBinder',
]
num_affs = len(aff_types)
aff_to_id = {v: k for k, v in enumerate(aff_types)}

# atom implicit valence
impv_types = list(range(7)) + ['other']
num_impv = len(impv_types)
impv_to_id = {v: k for k, v in enumerate(impv_types)}

# atom explicit valence
empv_types = list(range(1, 7)) + ['other']
num_empv = len(empv_types)
empv_to_id = {v: k for k, v in enumerate(empv_types)}


###################################### bond property
# get real ligand connectivity types
bond_types = [
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC',
    'other',
]
num_bontypes = len(bond_types)
bondtypes_to_id = {v: k for k, v in enumerate(bond_types)}

# get the geometry graph connectivity types
# compared to the :attr:`bond_types`, more than one
# called `NoneType` indicates that no chemical bond
# connected but in the 3D geometry, there is the
# geometric relation, such as angle constraints graph edge,
# dihedral constraints graph edge, ring constraints edge, etc
# without bond connectivity.
connect_types = [
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC',
    'other',
    'NoneType',
]
num_connecttypes = len(connect_types)
connect_to_id = {v: k for k, v in enumerate(connect_types)}

# the bond stereo types
bondstereo_types = [
    'STEREONONE',
    'STEREOANY',
    'STEREOZ',
    'STEREOE',
    'STEREOTRANS',
    'STEREOCIS',
]
num_bondstereos = len(bondstereo_types)
bondstereo_to_id = {v: k for k, v in enumerate(bondstereo_types)}


def one_hot_encoding(
        value: Union[str, int],
        value_types: Union[List[int], List[str]],
) -> List[int]:
    """
    Ligand one-hot encoding function.
    Note that we use `other` to distinguish wether allow unknown
        types. If the last elements of value_types are `other`, it
        means that any one not in the value_types will return the
        one-hot vector with the last one element of 1. Otherwise,
        it raise ValueError.
    Args:
        value: str or int. If `other` is in `value_types`, value
            can be any values. Otherwise, it must be the one of `values`
            in `value_types`.
        value_types: A list of int or str. one-hot encoding classes.
            return the one-hot encoding; otherwise, return the index value
    Returns:
        A one-hot encoding list.
    """
    encoding = [0] * len(value_types)
    if value in value_types:
        idx = value_types.index(value_types)
        encoding[idx] = 1
        return encoding
    if 'other' in value_types:
        encoding[-1] = 1
        return encoding
    else:
        raise ValueError(
            f'The input value `{value}` is not in `value_types`, '
            f'which does not allow other types.'
        )

def types_index(
        value: Union[str, int],
        value_types: Union[List[int], List[str]],
) -> int:
    """
    A simple function implementation to merge any unknown types
    A small version implementation with :func:`one_hot_encoding`.
        But return the index in :args:value_types.
    Args:
        value: str or int. If `other` is in `value_types`, value
            can be any values. Otherwise, it must be the one of `values`
            in `value_types`.
        value_types: A list of int or str. one-hot encoding classes.
            return the one-hot encoding; otherwise, return the index value
    Returns:
        The index of :args:`value` in :args:`value_types`.
    """
    if value in value_types:
        return value_types.index(value)
    if 'other' in value_types:
        return value_types.index('other')
    else:
        raise ValueError(
            f'The input value `{value}` is not in `value_types`, '
            f'which does not allow other types.'
        )

def _tree_key(data: Dict):
    if isinstance(data, dict):
        return {pt.GetAtomicNumber(k.capitalize()): _tree_key(v) for k, v in data.items()}
    else:
        return data

#### Force Filed parameters
# Van der Waals radii [Angstrom]
# reference:
#     1) https://en.wikipedia.org/wiki/Van_der_Waals_radius
#     2) Manjeera Mantina, et al. Consistent van der Waals Radii for the Whole Main Group.
#       J. Phys. Chem. A, 2009, 113, 19, 5806-5812.
vdW_radius = {
    'H': 1.10, 'He': 1.40,
    'Li': 1.81, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 2.75, 'Ca': 2.31, 'Sc': 2.11, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.83, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    'Cs': 3.43, 'Ba': 2.68, 'Pt': 1.75, 'Au': 1.66, 'Hg': 1.55, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
    'Fr': 3.48, 'Ra': 2.83, 'U': 1.86,
}
vdW_radius_pid = _tree_key(vdW_radius)

# Single-, double-, triple-bond covalent radii from https://en.wikipedia.org/wiki/Covalent_radius
single_covbr = {
    'H': 0.32, 'He': 0.46,
    'Li': 1.33, 'Be': 1.02, 'B': 0.85, 'C': 0.75, 'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67,
    'Na': 1.55, 'Mg': 1.39, 'Al': 1.26, 'Si': 1.16, 'P': 1.11, 'S': 1.03, 'Cl': 0.99, 'Ar': 0.96,
    'K': 1.96, 'Ca': 1.71, 'Sc': 1.48, 'Mn': 1.19, 'Fe': 1.16, 'Co': 1.11, 'Ni': 1.10, 'Cu': 1.12, 'Zn': 1.18, 'Ga': 1.24, 'Ge': 1.21,
    'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17,
    'Rb': 2.10, 'Sr': 1.85, 'Pd': 1.20, 'Ag': 1.28, 'Cd': 1.36, 'In': 1.42, 'Sn': 1.40, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
    'Cs': 2.32, 'Ba': 1.96, 'Pt': 1.23, 'Au': 1.24, 'Hg': 1.33, 'Tl': 1.44, 'Pb': 1.44, 'Bi': 1.51, 'Po': 1.45, 'At': 1.47, 'Rn': 1.42,
    'Fr': 2.23, 'Ra': 2.01, 'U': 1.70,
}

double_covbr = {
    'Li': 1.24, 'Be': 0.90, 'B': 0.78, 'C': 0.67, 'N': 0.60, 'O': 0.57, 'F': 0.59, 'Ne': 0.96,
    'Na': 1.60, 'Mg': 1.32, 'Al': 1.13, 'Si': 1.07, 'P': 1.02, 'S': 0.94, 'Cl': 0.95, 'Ar': 1.07,
    'K': 1.93, 'Ca': 1.47, 'Sc': 1.16, 'Mn': 1.05, 'Fe': 1.09, 'Co': 1.03, 'Ni': 1.01, 'Cu': 1.15, 'Zn': 1.20, 'Ga': 1.17, 'Ge': 1.11,
    'As': 1.14, 'Se': 1.07, 'Br': 1.09, 'Kr': 1.21,
    'Rb': 2.02, 'Sr': 1.57, 'Pd': 1.17, 'Ag': 1.39, 'Cd': 1.44, 'In': 1.36, 'Sn': 1.30, 'Sb': 1.33, 'Te': 1.28, 'I': 1.29, 'Xe': 1.35,
    'Cs': 2.09, 'Ba': 1.61, 'Pt': 1.12, 'Au': 1.21, 'Hg': 1.42, 'Tl': 1.42, 'Pb': 1.35, 'Bi': 1.41, 'Po': 1.35, 'At': 1.38, 'Rn': 1.45,
    'Fr': 2.18, 'Ra': 1.73, 'U': 1.34,
}

triple_covbr = {
    'Be': 0.85, 'B': 0.73, 'C': 0.60, 'N': 0.54, 'O': 0.53, 'F': 0.53,
    'Mg': 1.27, 'Al': 1.11, 'Si': 1.02, 'P': 0.94, 'S': 0.95, 'Cl': 0.93, 'Ar': 0.96,
    'Ca': 1.33, 'Sc': 1.14, 'Mn': 1.03, 'Fe': 1.02, 'Co': 0.96, 'Ni': 1.01, 'Cu': 1.20, 'Ga': 1.21, 'Ge': 1.14,
    'As': 1.06, 'Se': 1.07, 'Br': 1.10, 'Kr': 1.08,
    'Sr': 1.39, 'Pd': 1.12, 'Ag': 1.37, 'In': 1.46, 'Sn': 1.32, 'Sb': 1.27, 'Te': 1.21, 'I': 1.25, 'Xe': 1.22,
    'Ba': 1.61, 'Pt': 1.10, 'Au': 1.23, 'Tl': 1.50, 'Pb': 1.37, 'Bi': 1.35, 'Po': 1.29, 'At': 1.38, 'Rn': 1.33,
    'Ra': 1.59, 'U': 1.18,
}

single_covbr_pid = _tree_key(single_covbr)
double_covbr_pid = _tree_key(double_covbr)
triple_covbr_pid = _tree_key(triple_covbr)

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
@functools.lru_cache(maxsize = None)
def load_bond_length(
) -> Tuple[Dict[str, Dict[str, float]], ...]:
    bond_length = resources.read_text("druglib.resources", "bond_length.txt", encoding = 'gb18030')
    lines_iter = iter(bond_length.splitlines())
    single_bond_length = defaultdict(dict)
    double_bond_length = defaultdict(dict)
    triplet_bond_length = defaultdict(dict)

    def _search_atoms_and_bond(string: str):
        at1, at2 = re.split(r'[-,=,≡]', string)
        string = string[len(at1):]
        string = string[:-len(at2)]
        return at1, at2, string

    for line in lines_iter:
        bond, _, length = line.strip().split()
        at1, at2, bond = _search_atoms_and_bond(bond)
        # pm to angstrom
        length = float(length) / 100.
        if bond == '-':
            bond_dict = single_bond_length
        elif bond == '=':
            bond_dict = double_bond_length
        else:
            bond_dict = triplet_bond_length
        bond_dict[at1][at2] = length
        bond_dict[at2][at1] = length

    return single_bond_length, double_bond_length, triplet_bond_length

single_bond_length, double_bond_length, triplet_bond_length = load_bond_length()
single_bond_length_pid = _tree_key(single_bond_length)
double_bond_length_pid = _tree_key(double_bond_length)
triplet_bond_length_pid = _tree_key(triplet_bond_length)



