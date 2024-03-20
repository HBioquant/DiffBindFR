# Copyright (c) MDLDrugLib. All rights reserved.
import warnings
from pathlib import Path

import __main__
__main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI
import pymol2

from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning


def calc_centroid(
        sel: str,
        quite: bool = True,
        center: bool = False,
        cmd = None,
):
    """Calculate the centroid (geometric center) of a selection or move selection to origin."""
    from chempy import cpv
    if cmd is None:
        from pymol import cmd

    model = cmd.get_model(sel)
    nAtom = len(model.atom)

    centroid = cpv.get_null()

    for a in model.atom:
        centroid = cpv.add(centroid, a.coord)
    centroid = cpv.scale(centroid, 1. / nAtom)

    if not quite:
        print(' centroid: [%8.3f,%8.3f,%8.3f]' % tuple(centroid))

    if center:
        cmd.alter_state(1, sel, "(x,y,z)=sub((x,y,z), centroid)",
                        space={'centroid': centroid, 'sub': cpv.sub})

    return centroid

def parse_lig_center(
        ligfile: str,
        save: bool = False,
):
    ligfile = Path(ligfile)
    suffix = ligfile.suffix[1:]
    with pymol2.PyMOL() as pym:
        pymolcmd = pym.cmd
        pymolcmd.delete('all')
        # Use pymol to split protein and ligand
        pymolcmd.load(filename = str(ligfile), format=suffix, object='LIGCENTER')
        centroid = calc_centroid(sel = 'LIGCENTER', cmd=pymolcmd)
        pymolcmd.delete('all')

        centroid = ','.join([str(c) for c in centroid])
        if save:
            with open(str(ligfile.parent / ligfile.stem) + '_box.txt', 'w') as fout:
                fout.write(centroid)

    return centroid

def pdb_string_reader(
        id: str,
        pdbstring: str,
):
    """
    Biopython :cls.method:`Bio.PDB.PDBParser.get_structure` patcher.
    Convert pdb_string to Biopython object rather than loading from file.
    """
    protein = PDBParser(QUIET = 1)

    with warnings.catch_warnings():
        if protein.QUIET:
            warnings.filterwarnings("ignore", category = PDBConstructionWarning)

        protein.header = None
        protein.trailer = None
        # Make a StructureBuilder instance (pass id of structure as parameter)
        protein.structure_builder.init_structure(id)

        lines = pdbstring.split('\n')
        if not lines:
            raise ValueError("Empty string.")
        protein._parse(lines)

        protein.structure_builder.set_header(protein.header)
        # Return the Structure instance
        structure = protein.structure_builder.get_structure()

    return structure

def calc_sasa(
        prot_file: str,
        lig_file: str,
):
    pname = Path(prot_file).stem
    psuffix = Path(prot_file).suffix[1:]
    if pname == 'protein':
        pname = 'PROTID'
    lname = Path(lig_file).stem
    lsuffix = Path(lig_file).suffix[1:]
    if lname == 'ligand':
        lname = 'LIGID'

    def _sasa_biopython(pdbid, pdb_string):
        sr = ShrakeRupley()
        struct = pdb_string_reader(pdbid, pdb_string)
        sr.compute(struct, level="S")
        sasa = round(struct.sasa, 2)
        return sasa

    with pymol2.PyMOL() as pym:
        pymolcmd = pym.cmd
        pymolcmd.delete('all')
        pymolcmd.load(filename = prot_file, format = psuffix, object = pname)
        pymolcmd.load(filename = lig_file, format = lsuffix, object = lname)
        complex_pdb_str = pymolcmd.get_pdbstr('all')
        protein_pdb_str = pymolcmd.get_pdbstr(pname)
        ligand_pdb_str = pymolcmd.get_pdbstr(lname)
        csasa = _sasa_biopython(pname, complex_pdb_str)
        psasa = _sasa_biopython(pname, protein_pdb_str)
        lsasa = _sasa_biopython(pname, ligand_pdb_str)
        deltasasa = lsasa + psasa - csasa
        pymolcmd.delete('all')

    return deltasasa, csasa, lsasa, psasa