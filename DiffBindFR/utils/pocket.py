# Copyright (c) MDLDrugLib. All rights reserved.
import os.path
from typing import List, Optional, Union, Dict
import logging, tempfile, contextlib
import os.path as osp
from io import StringIO
from glob import glob
from collections import defaultdict
import numpy as np

from rdkit import Chem

import prody
# slience prody
prody.LOGGER._logger.setLevel(logging.INFO)
from prody import parsePDB, writePDB
from prody.atomic.select import Selection


@contextlib.contextmanager
def temp_pdb_file(prody_obj, **kwargs):
    """function that create temp file"""
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as pdb_file:
        writePDB(pdb_file.name, prody_obj, **kwargs)
        yield pdb_file.name

def get_ligand_code(
        chainid: str,
        resid: Optional[int] = None,
        resname: Optional[str] = None,
        altloc: Optional[str] = None,
):
    if all(x is None for x in [resid, resname]):
        raise ValueError('resname or resid must be assign one.')

    strings = [f'chain {chainid}']
    if resid is not None:
        strings.append(f'resnum {resid}')

    if resname is not None:
        strings.append(f'resname {resname}')

    if altloc is not None:
        strings.append(f'altloc {altloc}')

    return ' and '.join(strings)


def sdf2prody(
        sdffile: str,
        prot_max_resnum: int = 0,
        lig_chainid: str = 'X',
        resnum_gap_new_lig: int = 10,
        lig_default_resname: str = 'UNL',
) -> prody.AtomGroup:
    if not os.path.exists(sdffile):
        raise FileNotFoundError(sdffile)

    mol = Chem.SDMolSupplier(sdffile)[0]
    pdbblock = Chem.MolToPDBBlock(mol)
    lig_prody = prody.parsePDBStream(StringIO(pdbblock))
    atoms_number = lig_prody.numAtoms()
    ligand_resnum = prot_max_resnum + resnum_gap_new_lig
    lig_prody.setResnums([ligand_resnum] * atoms_number)
    lig_prody.setChids([lig_chainid] * atoms_number)

    ligand_code = get_ligand_code(
        chainid=lig_chainid,
        resid=ligand_resnum,
        resname=lig_default_resname,
    )

    lig_prody._data['ligand_code'] = ligand_code

    return lig_prody

def select_pocket(
        protein: prody.AtomGroup,
        ligand_code: str,
        dist_cutoff: float = 4.0,
) -> Selection:
    return protein.select(
        f'protein and within {dist_cutoff} of ({ligand_code})'
    )

def get_ligand_resnum(
        protein: prody.AtomGroup,
        ligand_code: str,
) -> int:
    lig_resnum = protein.select(ligand_code).getResnums()[0]
    return lig_resnum

def show_pocket_ligand(
        pdb: str,
        ligand_code: str,
        chainid: Optional[Union[str, List[str]]] = None,
        dist_cutoff: float = 4.0,
        view_parameters: dict = {
            "fogNear": 0, "fogFar": 100,
            "backgroundColor": "white",
        },
):
    """
    Args:
        pdb: PDBID or PDB file.
        ligand_code: [ligand chain id] [ligand residue number] [ligand resname] [altloc]
        chainid: any number of chains are allowed
        dist_cutoff: pocket sphere radius.
    """
    import nglview as nv

    prot = parsePDB(pdb)
    if chainid is not None:
        if isinstance(chainid, str):
            chainid = chainid.split(',')
        chainid = [x.strip() for x in chainid]
        chainid = f'chid {" ".join(chainid)}'
        prot = prot.select(chainid).toAtomGroup()

    pkt = select_pocket(
        protein = prot,
        ligand_code = ligand_code,
        dist_cutoff = dist_cutoff,
    )
    pkt_resnum_str = get_pocket_resnums_nv_str(pocket = pkt)
    lig_resnum = get_ligand_resnum(prot, ligand_code)
    view = nv.show_prody(prot)
    view.clear_representations()
    view.add_representation(
        'cartoon',
        selection = 'protein',
        color = 'orange')
    view.add_cartoon(
        selection = pkt_resnum_str,
        color = 'cyan')
    view.add_licorice(
        selection = pkt_resnum_str,
        color = 'cyan')
    view.add_representation(
        'spacefill',
        selection = f'{lig_resnum}')
    view.parameters = view_parameters
    return view

def get_pocket_resnums_dict(
        pocket: Selection,
) -> Dict[str, List[int]]:
    chain_number_dict = defaultdict(list)
    for res in pocket.getHierView().iterResidues():
        chid = res.getChid()
        resnum = res.getResnum()
        chain_number_dict[chid].append(resnum)
    return chain_number_dict

def resnum_dict_to_nv_str(
        chain_number_dict: Dict[str, List[int]]
) -> str:
    strings = []
    for k, v in chain_number_dict.items():
        v = list(map(str, v))
        if len(v) > 1:
            strings.append(f':{k} and ( {" or ".join(v)} )')
        else:
            strings.append(f':{k} and {v[0]}')

    if len(strings) == 1:
        return strings[0]

    strings = [f"( {x} )" for x in strings]
    return ' or '.join(strings)

def resnum_dict_to_prody_str(
        chain_number_dict: Dict[str, List[int]]
) -> str:
    strings = []
    for k, v in chain_number_dict.items():
        v = list(map(str, v))
        strings.append(f'chain {k} and resnum {" ".join(v)}')

    if len(strings) == 1:
        return strings[0]

    strings = [f"({x})" for x in strings]
    return ' or '.join(strings)

def resnum_dict_to_bsalign_str(
        chain_number_dict: Dict[str, List[int]]
) -> str:
    strings = []
    for k, v in chain_number_dict.items():
        strings.extend([f'{k}:{vv}' for vv in v])

    return ','.join(strings)

def get_pocket_resnums_nv_str(
        pocket: Selection,
) -> str:
    d = get_pocket_resnums_dict(pocket)
    return resnum_dict_to_nv_str(d)

def get_pocket_resnums_prody_str(
        pocket: Selection,
) -> str:
    d = get_pocket_resnums_dict(pocket)
    return resnum_dict_to_prody_str(d)

def get_pocket_resnums_bsalign_str(
        pocket: Selection,
) -> str:
    d = get_pocket_resnums_dict(pocket)
    return resnum_dict_to_bsalign_str(d)

class PDBPocketResidues:
    """
    Parse Holo ligand binding pocket residues.
    pdb: PDBID or PDB file.
    ligand_code: [ligand chain id] [ligand resindex] [ligand resname] [altloc]
    chainid: any number of chains are allowed
    dist_cutoff: pocket sphere radius.
    """
    def __init__(
            self,
            pdb: str,
            ligand_code: str,
            chainid: Optional[Union[str, List[str]]] = None,
    ):
        self.pdb = pdb
        self.ligand_code = ligand_code
        self.system = parsePDB(pdb)

        if chainid is not None:
            if isinstance(chainid, str):
                chainid = chainid.split(',')
            chainid = [x.strip() for x in chainid]
        self.chainid = chainid

        # load protein atomgroup
        if chainid is not None:
            self.protein = self.system.select(f'protein and chid {" ".join(chainid)}').toAtomGroup()
        else:
            self.protein = self.system.select(f'protein').toAtomGroup()

        # load heavy-atom ligand selection
        self.ligand = self.system.select(f'{ligand_code} and not element H')
        assert self.ligand is not None

        self.pocket_resnums_dict = None

    def get_pocket_resnums_dict(self, cutoff: float = 7.0):
        """
        Returns a list of residue numbers surrounding
        Args:
            cutoff (float): Distance cutoff to select the protein atoms surrounding the ligand
        Returns:
            pocket_residues (dict): A dict of residue numbers of the pocket mapped by chain
        """
        pocket_sel = self.protein.select(
            f'protein and same residue as exwithin {cutoff} of crypose',
            crypose = self.ligand.getCoords(),
        )
        assert pocket_sel is not None
        pocket_resnums = get_pocket_resnums_dict(pocket_sel)

        self.pocket_resnums_dict = pocket_resnums
        return pocket_resnums

    def visualize_pocket(self, cutoff: float = 7.0):
        import nglview as nv

        residues_dict = self.get_pocket_resnums_dict(cutoff = cutoff)
        str_residues = resnum_dict_to_prody_str(residues_dict)
        pocket_atoms = self.protein.select(str_residues).getIndices()

        # note that AtomGroup should be input of nv.show_prody
        view = nv.show_prody(self.protein)
        view.clear_representations()
        view.add_representation('cartoon', selection = 'protein', color = 'white')
        view.add_licorice(selection = pocket_atoms, color = 'red')
        view.add_cartoon(selection = pocket_atoms, color = 'red')

        # Load ligand as nglview object
        ligand_nv = nv.ProdyStructure(self.ligand)
        view.add_structure(ligand_nv)
        return view

    @classmethod
    def RDmolPocketResidues(
            cls,
            pdb: str,
            sdffile: str,
            chainid: Optional[Union[str, List[str]]] = None,
            lig_chainid: str = 'X',
            resnum_gap_new_lig: int = 10,
            lig_default_resname: str = 'UNL'
    ):
        prot_prody = prody.parsePDB(pdb)
        prot_max_resnum = prot_prody.getResnums().max()

        lig_prody = sdf2prody(
            sdffile,
            prot_max_resnum,
            lig_chainid,
            resnum_gap_new_lig,
            lig_default_resname,
        )
        ligand_code = lig_prody._data.pop('ligand_code')

        complex_prody = prot_prody + lig_prody

        with temp_pdb_file(complex_prody) as fp:
            return cls(
                pdb = fp,
                ligand_code = ligand_code,
                chainid = chainid,
            )

    def compare(
            self,
            pdb: str,
            chainid: Optional[Union[str, List[str]]] = None,
            ligand_sdf: Optional[str] = None,
            ligand_code: Optional[str] = None,
            pocket_cutoff: float = 5.0,
            lig_chainid: str = 'X',
            resnum_gap_new_lig: int = 10,
            lig_default_resname: str = 'UNL',
    ):
        system = prody.parsePDB(pdb)

        if chainid is not None:
            if isinstance(chainid, str):
                chainid = chainid.split(',')
            chainid = [x.strip() for x in chainid]

        if chainid is not None:
            protein = system.select(f'protein and chid {" ".join(chainid)}').toAtomGroup()
        else:
            protein = system.select(f'protein').toAtomGroup()

        ligand = None
        if ligand_sdf is not None:
            prot_max_resnum = protein.getResnums().max()
            ligand = sdf2prody(
                sdffile=ligand_sdf,
                prot_max_resnum=prot_max_resnum,
                lig_chainid=lig_chainid,
                resnum_gap_new_lig=resnum_gap_new_lig,
                lig_default_resname=lig_default_resname,
            )
            ligand_code = ligand._data.pop('ligand_code')
        elif ligand_code is not None:
            ligand = self.system.select(f'{ligand_code} and not element H')

        pocket_sel = protein.select(
            f'protein and same residue as exwithin {pocket_cutoff} of crypose',
            crypose=self.ligand.getCoords(),
        )
        assert pocket_sel is not None
        pocket_resnums = get_pocket_resnums_dict(pocket_sel)

        str_residues = resnum_dict_to_prody_str(pocket_resnums)
        pocket_atoms = protein.select(str_residues).getIndices()

        view = self.visualize_pocket(cutoff=pocket_cutoff)
        with temp_pdb_file(protein) as fp:
            v2 = view.add_component(fp)

        v2.clear()
        # v2.add_representation(repr_type='cartoon', selection='protein', colorScheme = 'sstruc')
        v2.add_representation('cartoon', selection='protein', color='yellow')
        v2.add_licorice(selection=pocket_atoms, color='blue')
        v2.add_cartoon(selection=pocket_atoms, color='blue')

        if ligand is not None:
            with temp_pdb_file(ligand) as fp:
                v3 = view.add_component(fp)
                v3.clear()
                ligname = ligand.getResnames()[0]
                v3.add_representation(
                    repr_type='licorice', radius=0.3,
                    color='green', selection=ligname,
                )

        return view

    def __repr__(self):
        residues_dict = self.get_pocket_resnums_dict(cutoff=5)
        str_residues = resnum_dict_to_prody_str(residues_dict)
        return f'{str_residues} within 5A of ligand'

def get_pocket_ligand(
        pdb_id: str,
        pocket_residues: str,
        raw_lig_dir: str,
        prot_chain_dir: str,
        pk_ligs_dir: str,
        cutoff: int = 3,
        min_weight: float = 97.0,
        write_files: bool = True,
        debug: bool = True,
):
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    lig_file = glob(f'{raw_lig_dir}/{pdb_id}*.pdb')
    if len(lig_file) == 0:
        logging.debug(f"The protein {pdb_id} have NO LIGAND.")
        return (None, None)
    lig = parsePDB(lig_file[0])

    prot_file = glob(f'{prot_chain_dir}/{pdb_id}*.pdb')
    if len(prot_file) == 0:
        logging.debug(f"Protein file not found: {pdb_id}")
        return (None, None)
    protein = parsePDB(prot_file[0])

    protein_pocket = protein.select("resnum " + pocket_residues)
    lig_sel = lig.select('within ' + str(cutoff) + ' of prot', prot = protein_pocket)
    if lig_sel is None:
        logging.debug(f"The protein {pdb_id} has no ligand inside the pocket.")
        return (None, None)

    inhibidor_list = np.unique(lig_sel.getResnames())
    # prot_pocket_center = calcCenter(protein_pocket)

    nearest_chain = ''
    nearest_resnum = ''
    nearest_resname = ''
    current_mass = 0
    for resname in inhibidor_list:
        mol = lig.select("resname " + resname)
        mol_chains = np.unique(mol.getChids())
        for chain in mol_chains:
            mol_resnums = np.unique(mol.select('chain ' + chain).getResnums())
            for resnum in mol_resnums:
                new_mol = mol.select(
                    'chain ' + chain + ' and resnum ' + str(resnum))
                new_mass = new_mol.getMasses().sum()

                if new_mass > current_mass:
                    # dist_new_mol = calcDistance(prot_pocket_center, calcCenter(new_mol))
                    nearest_chain = chain
                    nearest_resnum = str(resnum)
                    nearest_resname = resname
                    current_mass = new_mass

    true_lig = lig.select('chain ' + nearest_chain + \
                          ' and resnum ' + nearest_resnum + \
                          ' and resname ' + nearest_resname)
    lig_mass = true_lig.getMasses().sum()

    if true_lig != "" and lig_mass > min_weight:
        name_lig = np.unique(true_lig.getResnames())[0]
        complex_PL = protein + true_lig.toAtomGroup()
        if write_files:
            logging.debug(F'Protein {pdb_id}:\n   Molecules found: {str(inhibidor_list)} -> ligand {name_lig} saved.')
            writePDB(
                osp.join(pk_ligs_dir, pdb_id + "_" + name_lig + "_LIG.pdb"),
                complex_PL.select("resname " + name_lig))
        return (name_lig, lig_mass)
    else:
        logging.debug(F"The model {pdb_id} HAS NO LIGAND inside the pocket.")
        return (None, None)
