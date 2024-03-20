# Copyright (c) MDLDrugLib. All rights reserved.
import logging, operator
from typing import List, Dict, Tuple, Optional, Union
from string import Template
from collections import defaultdict

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import torch

import prody
from prody import parsePDB, parseMMCIF, calcDihedral
from Bio import pairwise2, PDB
from Bio.PDB.Polypeptide import one_to_three
from rdkit import Chem
from .io import exists_or_assert
from druglib.utils.obj.protein_constants import restype_name_to_atom14_names
from druglib.alerts import Error

class NoSequenceError(Error):
    """An error indicating that protein sequence doesn't exist by chain."""

class ProdyParsingError(Error):
    """An error indicating that pdb or mmcif file parsing failed."""

class NonHetSelect(PDB.Select):
    def accept_residue(self, residue):
        return 1 if PDB.Polypeptide.is_aa(residue,standard = True) else 0

def clean_pdb(
        input_file,
        output_file,
):
    """
    Takes a PDB file and removes hetero atoms from its structure.
    Args:
        input_file: path to original file
        output_file: path to generated file
    """
    pdb = PDB.PDBParser().get_structure("protein", input_file)
    io = PDB.PDBIO()
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())

def all_resindex(
        prot_ag,
) -> Dict[str, List[int]]:
    """Map chains to residue and residue index"""
    chains = prot_ag.calpha.getChids()
    resind = prot_ag.calpha.getResindices()
    resindex_map, ch_cur = defaultdict(list), None
    for ch, ind in zip(chains, resind):
        if ch == ' ': continue # ions
        if ch != ch_cur:
           ch_cur = ch
        resindex_map[ch_cur].append(ind)

    return resindex_map

def selected_resindex(
        prot_ag,
        sel_chains: List[str],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    # residue indices <-> sequence <-> chains
    # selected chains and summary of atom residue, chain, chain len
    seq, chseq, chptr = [], [], {}
    for ch in sel_chains:
        if ch == ' ': continue # ions
        else:
            try:
                # sometimes pymol gives chains not really involved in alignment
                _seq = prot_ag.select(f'protein and chain {ch}').calpha.getSequence()
                chseq += [ch] * len(_seq)
                chptr[ch] = len(seq)
                seq += _seq
            except: continue

    return seq, chseq, chptr

def seq_align(
        holo_seq: List[str],
        apo_seq: List[str],
) -> Tuple[str, str]:
    holo_seq = ''.join(holo_seq)
    apo_seq = ''.join(apo_seq)
    if not (holo_seq and apo_seq):
        raise NoSequenceError

    alignment = pairwise2.align.globalxx(
        holo_seq, apo_seq
    )[0]
    holo_seq = alignment[0]
    apo_seq = alignment[1]

    return holo_seq, apo_seq

def get_indices(
        alignment: str,
) -> List[Union[int, str]]:
    # str to list
    alignment = list(alignment)
    indices, count = [], 0
    for i in alignment:
        if i == '-': indices.append(i)
        else:
            indices.append(count)
            count += 1

    return indices

def TMScore(
        holo_prody, apo_prody,
        holo_aligenment, apo_alignment,
        holo_align_ind, apo_align_ind,
        holo_chseq, apo_chseq,
        holo_chptr, apo_chptr,
        holo_crmap, apo_crmap,
) -> float:
    holo_cas, apo_cas = [], []
    L = len(holo_chseq)
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    d02 = d0 ** 2
    assert len(holo_aligenment) == len(apo_alignment)
    for i in range(len(holo_aligenment)):
        holo_res = holo_aligenment[i]
        if holo_res == '-': continue
        apo_res = apo_alignment[i]
        if holo_res == apo_res:
            holo_seq_ind = holo_align_ind[i]
            apo_seq_ind = apo_align_ind[i]
            holo_chid = holo_chseq[holo_seq_ind]
            apo_chid = apo_chseq[apo_seq_ind]
            holo_res_ind = holo_crmap[holo_chid][
                holo_seq_ind - holo_chptr[holo_chid]]
            apo_res_ind = apo_crmap[apo_chid][
                apo_seq_ind - apo_chptr[apo_chid]]
            holo_ca = holo_prody.select(
                'chain ' + str(holo_chid) + ' and ' + 'resindex ' + str(holo_res_ind) + ' and name CA')
            apo_ca = apo_prody.select(
                'chain ' + str(apo_chid) + ' and ' + 'resindex ' + str(apo_res_ind) + ' and name CA')
            # sometimes more than 1 calpha present,
            # take care by taking only the first occurence
            holo_ca = holo_ca.getCoords()[0]
            holo_cas.append(holo_ca)
            apo_ca = apo_ca.getCoords()[0]
            apo_cas.append(apo_ca)
    assert len(holo_cas) == len(apo_cas)
    holo_cas = np.array(holo_cas)
    apo_cas = np.array(apo_cas)
    d1s = np.sum(np.square(holo_cas - apo_cas), axis = 1)
    tmarray = 1 / (1 + np.divide(d1s, d02))
    tmscore = (1 / L) * np.sum(tmarray)

    return tmscore

def bs_res_finder(
        ligand_coords: np.ndarray,
        holo_prody: prody.AtomGroup,
        bs_cutoff: float = 6.0,
):
    holo_coords = holo_prody.getCoords()
    holo_all_chs = holo_prody.getChids()
    holo_all_ress = holo_prody.getResindices()
    holo_all_resn = holo_prody.getResnames()
    ligand_dist = cdist(ligand_coords, holo_coords)
    bs_indices = np.where(np.any(ligand_dist <= bs_cutoff, axis=0))
    holo_bindsite = []
    for c, r, n in zip(holo_all_chs[bs_indices], holo_all_ress[bs_indices], holo_all_resn[bs_indices]):
        if c == ' ': continue
        if (c, r, n) in holo_bindsite: continue
        holo_bindsite.append((c, r, n))

    return holo_bindsite

def specific_bs_res(
        holo_prody: prody.AtomGroup,
        bs_res_str: List[str], # chain:residue number:residue name
):
    res_sel = []
    for x in bs_res_str:
        xs = x.split(':')
        res_sel.append(f'(chain {xs[0]} and resnum {xs[1]} and resname {xs[2]})')

    res_sel = ' or '.join(res_sel)
    pkt = holo_prody.select(res_sel)
    holo_bindsite = []
    if pkt is None:
        return holo_bindsite
    for res in pkt.getHierView().iterResidues():
        c, r, n = res.getChid(), res.getResindex(), res.getResname()
        if c == ' ': continue
        if (c, r, n) in holo_bindsite: continue
        holo_bindsite.append((c, r, n))

    return holo_bindsite

def get_belongs(
        value,
        obj2ptr,
):
    diff = np.inf
    obj = None
    for o in obj2ptr.keys():
        if value >= obj2ptr[o]:
            _diff = value - obj2ptr[o]
            if _diff < diff:
                diff = _diff
                obj = o
    return obj

def get_apo_bind_indices(
        holo_bindsite,
        holo_seq, apo_seq,
        holo_align_ind, apo_align_ind,
        holo_chptr, apo_chptr,
        holo_crmap, apo_crmap,
) -> Tuple[List[int], List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    apo_holo_mask, apo_bind_res, holo_bind_res = [], [], []
    for (holo_chain, holo_res_ind, holo_res_name) in holo_bindsite:
        holo_seq_ind = holo_chptr[holo_chain] + \
                       holo_crmap[holo_chain].index(holo_res_ind)
        holo_aln_index = holo_align_ind.index(holo_seq_ind)
        holo_res = holo_seq[holo_seq_ind]
        assert holo_res == prody.atomic.AAMAP.get(holo_res_name, 'X')
        # holo binding site chain id, res one-letter name, residue index
        holo_bind_res.append((holo_chain, holo_res, holo_res_ind))
        apo_seq_index = apo_align_ind[holo_aln_index]
        if apo_seq_index == '-':
            apo_bind_res.append(('-', '-', '-'))
            apo_holo_mask.append(0)
            continue
        apo_res = apo_seq[apo_seq_index]
        apo_chain = get_belongs(apo_seq_index, apo_chptr)
        apo_res_ind = apo_crmap[apo_chain][apo_seq_index - apo_chptr[apo_chain]]
        apo_bind_res.append((apo_chain, apo_res, apo_res_ind))
        #  if the residue pair match
        if apo_res == holo_res:
            apo_holo_mask.append(1)
        else:
            apo_holo_mask.append(0)

    return apo_holo_mask, holo_bind_res, apo_bind_res

def chain_selection(chid: Union[str, List[str]]):
    if isinstance(chid, str):
        chid = chid.split(',')
    return ' and '.join([f'chain {i}' for i in chid])

class ApoHoloBS:
    chi_atoms = dict(
        chi1=dict(
            ARG=['N', 'CA', 'CB', 'CG'],
            ASN=['N', 'CA', 'CB', 'CG'],
            ASP=['N', 'CA', 'CB', 'CG'],
            CYS=['N', 'CA', 'CB', 'SG'],
            GLN=['N', 'CA', 'CB', 'CG'],
            GLU=['N', 'CA', 'CB', 'CG'],
            HIS=['N', 'CA', 'CB', 'CG'],
            ILE=['N', 'CA', 'CB', 'CG1'],
            LEU=['N', 'CA', 'CB', 'CG'],
            LYS=['N', 'CA', 'CB', 'CG'],
            MET=['N', 'CA', 'CB', 'CG'],
            PHE=['N', 'CA', 'CB', 'CG'],
            PRO=['N', 'CA', 'CB', 'CG'],
            SER=['N', 'CA', 'CB', 'OG'],
            THR=['N', 'CA', 'CB', 'OG1'],
            TRP=['N', 'CA', 'CB', 'CG'],
            TYR=['N', 'CA', 'CB', 'CG'],
            VAL=['N', 'CA', 'CB', 'CG1'],
        ),
        altchi1=dict(
            VAL=['N', 'CA', 'CB', 'CG2'],
        ),
        chi2=dict(
            ARG=['CA', 'CB', 'CG', 'CD'],
            ASN=['CA', 'CB', 'CG', 'OD1'],
            ASP=['CA', 'CB', 'CG', 'OD1'],
            GLN=['CA', 'CB', 'CG', 'CD'],
            GLU=['CA', 'CB', 'CG', 'CD'],
            HIS=['CA', 'CB', 'CG', 'ND1'],
            ILE=['CA', 'CB', 'CG1', 'CD1'],
            LEU=['CA', 'CB', 'CG', 'CD1'],
            LYS=['CA', 'CB', 'CG', 'CD'],
            MET=['CA', 'CB', 'CG', 'SD'],
            PHE=['CA', 'CB', 'CG', 'CD1'],
            PRO=['CA', 'CB', 'CG', 'CD'],
            TRP=['CA', 'CB', 'CG', 'CD1'],
            TYR=['CA', 'CB', 'CG', 'CD1'],
        ),
        altchi2=dict(
            ASP=['CA', 'CB', 'CG', 'OD2'],
            LEU=['CA', 'CB', 'CG', 'CD2'],
            PHE=['CA', 'CB', 'CG', 'CD2'],
            TYR=['CA', 'CB', 'CG', 'CD2'],
        ),
        chi3=dict(
            ARG=['CB', 'CG', 'CD', 'NE'],
            GLN=['CB', 'CG', 'CD', 'OE1'],
            GLU=['CB', 'CG', 'CD', 'OE1'],
            LYS=['CB', 'CG', 'CD', 'CE'],
            MET=['CB', 'CG', 'SD', 'CE'],
        ),
        chi4=dict(
            ARG=['CG', 'CD', 'NE', 'CZ'],
            LYS=['CG', 'CD', 'CE', 'NZ'],
        ),
        chi5=dict(
            ARG=['CD', 'NE', 'CZ', 'NH1'],
        ),
    )
    sel_res_template = Template(
        "chain $chid and resindex $residx "
    )
    sel_sc_template = Template(
        "chain $chid and resindex $residx "
        "and not element H and not name CA "
        "and not name C and not name O and not name N "
    )
    sel_res_atom_template = Template(
        "chain $chid and resindex $residx and name $atom "
    )
    default_chi = [1]
    def __init__(
            self,
            apo: prody.AtomGroup,
            holo: prody.AtomGroup,
            aln_apo_residues: List[Tuple[str, str, int]],
            aln_holo_residues: List[Tuple[str, str, int]],
            pair_flag: List[Union[int, bool]],
            chi: Optional[Union[int, List[int]]] = None,
            units: Optional[str] = None,
            debug: bool = True,
    ):
        self.apo = apo
        self.holo = holo
        assert len(aln_holo_residues) == len(aln_apo_residues) and len(aln_holo_residues) == len(pair_flag)
        self.aln_apo_residues = aln_apo_residues
        self.aln_holo_residues = aln_holo_residues
        self.pair_mask = pair_flag
        self.logger = logging.getLogger(self.__class__.__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.set_chi(chi)
        self.return_degree(units)

        self.run()

    def set_chi(self, chi) -> None:
        if chi is None:
            chi = self.default_chi
        if isinstance(chi, int):
            chi = [chi]

        chi_names = list()
        for x in chi:
            reg_chi = "chi%s" % x
            if reg_chi in self.chi_atoms.keys():
                chi_names.append(reg_chi)
                alt_chi = "altchi%s" % x
                if alt_chi in self.chi_atoms.keys():
                    chi_names.append(alt_chi)
            else:
                self.logger.warning("Invalid chi %s", x)

        self.chi_names = chi_names
        self.logger.info(f'Calculate chi {",".join(chi_names)}.')

    def return_degree(self, units) -> None:
        if units is None:
            units = "degrees"
        self.degrees = bool(units[0].lower() == "d")
        if self.degrees:
            message = "Using degrees"
        else:
            message = "Using radians"
        self.logger.debug(message)

    def residue_to_arom14_array(
            self,
            prody_obj: prody.AtomGroup,
            chid: str,
            resn: str,
            resindx: int,
            _prot_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if _prot_type is None:
            _prot_type = ''
        else:
            _prot_type = _prot_type + ' '
        atom14_pos = np.zeros((14, 3), dtype = float)
        atom14_mask  = np.zeros((14, ), dtype = bool)
        resn3 = one_to_three(resn)
        if resn3 in restype_name_to_atom14_names:
            for idx, at in enumerate(restype_name_to_atom14_names[resn3]):
                if not at: continue
                ag = prody_obj.select(
                    self.sel_res_atom_template.substitute(
                        chid=chid,
                        residx=resindx,
                        atom=at,
                    )
                )
                if ag is None:
                    self.logger.debug(f'{_prot_type}ATOM14::: {chid}/{resindx}/{resn3}/{at} is missing')
                    continue
                atom14_pos[idx] = ag.getCoords()[0]
                atom14_mask[idx] = True

        return atom14_pos, atom14_mask

    def bs_atom14(
            self,
            prody_obj: prody.AtomGroup,
            aln_residues: List[Tuple[str, str, int]],
            _prot_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        atom14_pos, atom14_mask = [], []
        for map_flag, _res in zip(self.pair_mask, aln_residues):
            if not map_flag: continue
            pos, mask = self.residue_to_arom14_array(
                prody_obj, *_res, _prot_type = _prot_type
            )
            atom14_pos.append(pos)
            atom14_mask.append(mask)

        if len(atom14_pos) > 0:
            atom14_pos = np.stack(atom14_pos, axis = 0)
            atom14_mask = np.stack(atom14_mask, axis = 0)
        else:
            atom14_pos = np.zeros((0, 14, 3), dtype = float)
            atom14_mask = np.zeros((0, 14), dtype = bool)

        return atom14_pos, atom14_mask

    def atom14(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        self.holo_atom14 = self.bs_atom14(
            self.holo, self.aln_holo_residues, 'Holo',
        )
        self.apo_atom14 = self.bs_atom14(
            self.apo, self.aln_apo_residues, 'Apo',
        )
        return self.holo_atom14, self.apo_atom14

    def run(
            self,
            lig_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
            **kwargs,
    ):
        self.calculate_sc()
        self.calculate_ca()
        self.calculate_chi()
        if lig_coords is not None:
            self.calculate_plddt_pli(lig_coords, **kwargs)

    def calculate_sc(self) -> Tuple[float, List[float]]:
        per_sc_rmsd, mean_sc_rmsd, count = [np.NaN] * sum(self.pair_mask), [], -1
        for map_flag, holo_res, apo_res in zip(
                self.pair_mask,
                self.aln_holo_residues,
                self.aln_apo_residues,
        ):
            if not map_flag: continue
            count += 1
            holo_ag = self.holo.select(
                self.sel_sc_template.substitute(
                    chid=holo_res[0],
                    residx=holo_res[2],
                ),
            )
            apo_ag = self.apo.select(
                self.sel_sc_template.substitute(
                    chid=apo_res[0],
                    residx=apo_res[2],
                )
            )
            if holo_ag is None or apo_ag is None:
                if holo_res not in ['A', 'G']:
                    self.logger.debug(f'SC Calculator::: '
                                      f'holo ag ({"/".join([str(i) for i in holo_res])}) and '
                                      f'apo ag ({"/".join([str(i) for i in apo_res])}) '
                                      f'side chain missing')
                continue
            holo_ag_names = holo_ag.getNames()
            apo_ag_names = apo_ag.getNames()
            # check for missing atoms/non-standard residues
            if len(holo_ag_names) != len(apo_ag_names):
                self.logger.debug(f'SC Calculator::: '
                                  f'atom group mismatch: '
                                  f'holo_ag_names ({"/".join([str(i) for i in holo_res])}, {len(holo_ag_names)}); '
                                  f'apo_ag_names ({"/".join([str(i) for i in apo_res])}, {len(apo_ag_names)})')
                continue
            enumerate_holo = enumerate(holo_ag_names)
            holo_sorted_pairs = sorted(
                enumerate_holo, key=operator.itemgetter(1))
            enumerate_apo = enumerate(apo_ag_names)
            apo_sorted_pairs = sorted(
                enumerate_apo, key=operator.itemgetter(1))
            holo_sorted_indices, apo_sorted_indices = [], []
            for index, element in holo_sorted_pairs:
                holo_sorted_indices.append(index)
            for index, element in apo_sorted_pairs:
                apo_sorted_indices.append(index)
            _sc_dist = np.sum(
                (holo_ag.getCoords()[holo_sorted_indices] -
                 apo_ag.getCoords()[apo_sorted_indices]) ** 2,
                axis=1,
            )
            per_sc_rmsd[count] = np.sqrt(_sc_dist.mean())
            mean_sc_rmsd.append(_sc_dist)

        self.per_sc_rmsd = per_sc_rmsd
        if len(mean_sc_rmsd) > 0:
            self.mean_sc_rmsd = np.sqrt(np.concatenate(mean_sc_rmsd, axis=0).mean())
        else:
            self.logger.error(f'{sum(self.pair_mask)} matched pocket residues have No sc rmsd.')
            self.mean_sc_rmsd = np.NaN

        return self.mean_sc_rmsd, self.per_sc_rmsd

    def calculate_chi(self):
        self.holo_per_chi = self.chi_fn(self.holo, self.aln_holo_residues, 'Holo')
        self.apo_per_chi = self.chi_fn(self.apo, self.aln_apo_residues, 'Apo')
        return self.holo_per_chi, self.apo_per_chi

    def chi_fn(
            self,
            prody_obj: prody.AtomGroup,
            aln_residues: List[Tuple[str, str, int]],
            _prot_type: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        if _prot_type is None:
            _prot_type = ''
        else:
            _prot_type = _prot_type + ' '

        per_chi = []
        for map_flag, _res in zip(self.pair_mask, aln_residues):
            if not map_flag: continue
            chi_list = [np.NaN] * len(self.chi_names)
            for atid, chi in enumerate(self.chi_names):
                chi_res_dict = self.chi_atoms[chi]
                try:
                    resn = one_to_three(_res[1])
                    atom_list = chi_res_dict[resn]
                    atom4 = []
                    for at in atom_list:
                        ag = prody_obj.select(
                            self.sel_res_atom_template.substitute(
                                chid=_res[0],
                                residx=_res[2],
                                atom=at,
                            ),
                        )
                        if ag is None:
                            self.logger.debug(f'{_prot_type}Chi Calculator::: '
                                              f'({"/".join([str(i) for i in _res])}, {chi}): '
                                              f'{atom_list} found {at} is missing.')
                            raise KeyError(f'{at} is missing')
                        atom4.append(ag)
                except KeyError:
                    continue

                angle = calcDihedral(*atom4, radian = not self.degrees)
                chi_list[atid] = angle[0]
            per_chi.append(chi_list)

        per_chi = list(zip(*per_chi))
        per_chi = dict(zip(self.chi_names, per_chi))
        return per_chi

    def calculate_ca(self) -> Tuple[float, List[float]]:
        per_ca_rmsd, mean_ca_rmsd, count = [np.NaN] * sum(self.pair_mask), [], -1
        for map_flag, holo_res, apo_res in zip(
                self.pair_mask,
                self.aln_holo_residues,
                self.aln_apo_residues,
        ):
            if not map_flag: continue
            count += 1
            holo_ag = self.holo.select(
                self.sel_res_atom_template.substitute(
                    chid=holo_res[0],
                    residx=holo_res[2],
                    atom='CA'
                ),
            )
            apo_ag = self.apo.select(
                self.sel_res_atom_template.substitute(
                    chid=apo_res[0],
                    residx=apo_res[2],
                    atom='CA'
                )
            )
            if holo_ag is None or apo_ag is None:
                self.logger.debug(f'CA Calculator::: '
                                  f'holo ag ({"/".join([str(i) for i in holo_res])}) and '
                                  f'apo ag ({"/".join([str(i) for i in apo_res])}) CA missing')
                continue
            _ca_dist = np.sum(
                (holo_ag.getCoords()[0] -
                 apo_ag.getCoords()[0]) ** 2,
            )
            per_ca_rmsd[count] = np.sqrt(_ca_dist)
            mean_ca_rmsd.append(_ca_dist)

        self.per_ca_rmsd = per_ca_rmsd

        if len(mean_ca_rmsd) > 0:
            self.ca_rmsd = np.sqrt(np.mean(mean_ca_rmsd))
        else:
            self.logger.error(f'{sum(self.pair_mask)} matched pocket residues have No CA rmsd.')
            self.ca_rmsd = np.NaN

        return self.ca_rmsd, self.per_ca_rmsd

    def calculate_plddt_pli(
            self,
            ligand_coords: Union[np.ndarray, torch.Tensor],
            lddt_inclusion_radius: float = 6.0,
            eps: float = 1e-9,
    ) -> Tuple[float, List[float]]:
        # (N_l, 3)
        if isinstance(ligand_coords, np.ndarray):
            ligand_coords = torch.from_numpy(ligand_coords).float()
        elif isinstance(ligand_coords, torch.Tensor):
            ligand_coords = ligand_coords.float()
        else:
            raise TypeError('Expect np.ndarray or torch.Tensor')

        def to_torch(atom14):
            pos, mask = atom14
            return torch.from_numpy(pos).float(), torch.from_numpy(mask).bool()

        # (N, 14, 3), (N, 14)
        holo_atom14, apo_atom14 = self.atom14()
        holo_atom14_pos, holo_atom14_mask = to_torch(holo_atom14)
        apo_atom14_pos, apo_atom14_mask = to_torch(apo_atom14)

        # (*, N_res, 14, 3) * (*, N_l, 3) -> (*, N_res, 14, N_l)
        cdmat_holo = torch.sqrt(
            eps + torch.sum(
                (holo_atom14_pos[..., None, :] - ligand_coords[..., None, None, :, :]) ** 2,
                dim=-1,
            )
        )
        # aligned apo
        cdmat_apo = torch.sqrt(
            eps + torch.sum(
                (apo_atom14_pos[..., None, :] - ligand_coords[..., None, None, :, :]) ** 2,
                dim=-1,
            )
        )
        # use ground truth distance map as score
        dists_to_score = (
                (cdmat_holo < lddt_inclusion_radius)
                * (holo_atom14_mask * apo_atom14_mask)[..., None]
        )

        dist_l1 = torch.abs(cdmat_holo - cdmat_apo)
        score = (
                (dist_l1 < 0.5).type(dist_l1.dtype)
                + (dist_l1 < 1.0).type(dist_l1.dtype)
                + (dist_l1 < 2.0).type(dist_l1.dtype)
                + (dist_l1 < 4.0).type(dist_l1.dtype)
        )
        score = score * 0.25

        # residue-wise (*, N_res)
        res_norm = 1.0 / (eps + torch.sum(dists_to_score, dim=(-2, -1)))
        res_score = res_norm * (eps + torch.sum(dists_to_score * score, dim=(-2, -1)))
        self.per_res_score = res_score.tolist()
        self.mean_res_score = res_score.mean().item()

        return self.mean_res_score, self.per_res_score

    @classmethod
    def get_resnums(
            cls,
            prody_obj: prody.AtomGroup,
            aln_residues: List[Tuple[str, str, int]],
    ) -> List[int]:
        aln_resnums = []
        for idx, _res in enumerate(aln_residues):
            if _res[2] == '-' or _res[1] == '-':
                aln_resnums.append('-')
                continue
            ag = prody_obj.select(
                cls.sel_res_template.substitute(
                    chid=_res[0],
                    residx=_res[2],
                ),
            )
            resnum = ag.getResnums()[0]
            aln_resnums.append(resnum)

        return aln_resnums

    def resnums(self) -> Tuple[List[int], List[int]]:
        self.aln_holo_resnums = self.get_resnums(self.holo, self.aln_holo_residues)
        self.aln_apo_resnums = self.get_resnums(self.apo, self.aln_apo_residues)

        return self.aln_holo_resnums, self.aln_apo_resnums

    @staticmethod
    def to_pymol_strings(
            aln_residues: List[Tuple[str, str, int]],
            aln_resnums: List[int],
    ) -> List[str]:
        pymol_strings = []
        for res, resnum in zip(aln_residues, aln_resnums):
            res = list(map(str, res))
            res[1] = one_to_three(res[1]) if res[1] != '-' else '-'
            pymol_strings.append('/'.join([''] + res + [str(resnum)]))

        return pymol_strings

    def export_pymol_strings(self) -> Tuple[List[str], List[str]]:
        self.resnums()
        holo_pymol_strings = self.to_pymol_strings(self.aln_holo_residues, self.aln_holo_resnums)
        apo_pymol_strings = self.to_pymol_strings(self.aln_apo_residues, self.aln_apo_resnums)
        return holo_pymol_strings, apo_pymol_strings

    def summary(self) -> pd.DataFrame:
        holo_pymol_strings, apo_pymol_strings = self.export_pymol_strings()
        export_df = defaultdict(list)
        for map_flag, holo_str, apo_str in zip(
                self.pair_mask,
                holo_pymol_strings,
                apo_pymol_strings,
        ):
            if not map_flag: continue
            export_df['holo_res'].append(holo_str)
            export_df['apo_res'].append(apo_str)

        # add prefix
        holo_per_chi = {('holo_' + k): v for k, v in self.holo_per_chi.items()}
        apo_per_chi = {('apo_' + k): v for k, v in self.apo_per_chi.items()}
        export_df.update(holo_per_chi)
        export_df.update(apo_per_chi)
        export_df['per_ca_rmsd'] = self.per_ca_rmsd
        export_df['mean_ca_rmsd'] = [self.ca_rmsd] * sum(self.pair_mask)
        export_df['per_sc_rmsd'] = self.per_sc_rmsd
        export_df['mean_sc_rmsd'] = [self.mean_sc_rmsd] * sum(self.pair_mask)
        export_df['per_plddt_pli'] = self.per_res_score
        export_df['mean_plddt_pli'] = [self.mean_res_score] * sum(self.pair_mask)
        if hasattr(self, 'tmscore'):
            export_df['tmscore'] = [self.tmscore] * sum(self.pair_mask)

        return pd.DataFrame(export_df)

    def set_tmscore(self, tmscore: float):
        self.tmscore = tmscore

def parse_fn(f):
    if f.endswith('.pdb') or f.endswith('.pdb.gz'):
        return parsePDB(f)
    elif f.endswith('.cif') or f.endswith('.cif.gz'):
        return parseMMCIF(f)
    else:
        raise NotImplementedError('Supported protein file: .pdb or .cif')

def pair_spatial_metrics(
        holo_pdb: str,
        lig_file: str,
        apo_pdb: str,
        holo_chains: str,
        apo_chains: str,
        return_bs: bool = False,
        bs_cutoff: float = 6.0,
        chi: Optional[Union[int, List[int]]] = [1, 2, 3, 4],
        units: Optional[str] = None,
        debug: bool = False,
        bs_res_str: Optional[List[str]] = None,
) -> Union[ApoHoloBS, pd.DataFrame]:
    """
    Calculate apo-holo pair spatial metrics,
        such as paired-residues RMSD, tmscore.
    Also return the binding site residues and indices.
    Args:
        holo_pdb: holo protein file.
        lig_file: ligand sdf file.
        apo_pdb: apo protein file.
        holo_chains: holo protein paired chains. formatting: 'A,B,C' or 'A'
        apo_chains: apo protein paired chains.
        bs_res_str: a list of residue selection expression as chain:residue number:residue name, optional
    """
    exists_or_assert(holo_pdb)
    exists_or_assert(lig_file)
    exists_or_assert(apo_pdb)
    holo_chains = [r.strip() for r in holo_chains.split(',')]
    apo_chains = [r.strip() for r in apo_chains.split(',')]

    holo_prody = parse_fn(holo_pdb).select(f'protein and {chain_selection(holo_chains)}')
    apo_prody = parse_fn(apo_pdb).select(f'protein and {chain_selection(apo_chains)}')
    if holo_prody is None or apo_prody is None:
        logging.error(
            f'{holo_pdb} by chain {holo_chains} prody object: {holo_prody}; '
            f'{apo_pdb} by chain {apo_chains} prody object: {apo_prody}.'

        )
        raise ProdyParsingError

    ## rcmap: chain -> resindex mapping
    holo_crmap = all_resindex(holo_prody)
    apo_crmap = all_resindex(apo_prody)
    holo_seq, holo_chseq, holo_chptr = selected_resindex(
        holo_prody, holo_chains,
    )
    apo_seq, apo_chseq, apo_chptr = selected_resindex(
        apo_prody, apo_chains,
    )
    # align sequence of holo and apo to calculate TMScore
    try:
        holo_aln_seq, apo_aln_seq = seq_align(
            holo_seq, apo_seq
        )
    except NoSequenceError:
        logging.error(
            f'{holo_pdb} by chain {holo_chains}: sequence {"".join(holo_seq)}; '
            f'{apo_pdb} by chain {apo_chains}: sequence {"".join(apo_seq)}.'
        )
        raise NoSequenceError

    holo_aln_ind = get_indices(holo_aln_seq)
    apo_aln_ind = get_indices(apo_aln_seq)
    tmscore = TMScore(
        holo_prody, apo_prody,
        holo_aln_seq, apo_aln_seq,
        holo_aln_ind, apo_aln_ind,
        holo_chseq, apo_chseq,
        holo_chptr, apo_chptr,
        holo_crmap, apo_crmap,
    )
    # load ligand
    mol = Chem.SDMolSupplier(lig_file, sanitize=False)[0]
    ligand_coords = mol.GetConformer().GetPositions()
    # load binding residues and its chains
    if bs_res_str is None:
        holo_bindsite = bs_res_finder(ligand_coords, holo_prody, bs_cutoff)
    else:
        holo_bindsite = specific_bs_res(holo_prody, bs_res_str)

    # map to alignment and sequence and back to apo_binding residues
    apo_holo_mask, holo_bind_res, apo_bind_res = get_apo_bind_indices(
        holo_bindsite,
        holo_seq, apo_seq,
        holo_aln_ind, apo_aln_ind,
        holo_chptr, apo_chptr,
        holo_crmap, apo_crmap,
    )
    ah_bs = ApoHoloBS(
        apo = apo_prody,
        holo = holo_prody,
        aln_apo_residues = apo_bind_res,
        aln_holo_residues = holo_bind_res,
        pair_flag = apo_holo_mask,
        chi = chi,
        units = units,
        debug = debug,
    )
    ah_bs.run(ligand_coords)
    ah_bs.set_tmscore(tmscore)

    if return_bs:
        return ah_bs

    return ah_bs.summary()

