# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Dict, Any
from functools import partial
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB


def fix_protein(
        filename: str,
        output: str,
        add_missing: bool = True,
        hydrogenate: bool = True,
        addHs_pH: float = 7.4,
        remove_heterogens: bool = True,
        replace_nonstandard_aa: bool = True,
        renumber_Residues: bool = False,
        return_info: bool = False,
):
    """
    Fix protein w.r.t missing residues, heterogens and atoms.
    Args:
        filename: str. Protein file to clean up.
        output: str. Fixed protein file.
        add_missing: bool, optional. If True,  add missing residues and atoms.
            Default to True.
        hydrogenate: bool, optional. If True, add hydrogens at specific pH.
            Default to True.
        addHs_pH: float, optional. The pH at which hydrogens will be added
            if `hydrogenate==True`. Set to 7.4 by default.
        remove_heterogens: bool, optional. pdb files usually attach with
            extra water and salt. If True, remove those.
        replace_nonstandard_aa: bool, optional. Replace the nonstandard amino acid.
            Default to True.
        renumber_Residues: bool, optional. After all the above pipelines, the amino acid
            number would change when save to output file. If True, keep orignal pdb AA number.
        return_info： bool, optional. Return protein fix details.
    Returns: info_dict. meta info about the fix info.
        Including: the fixed pdb file saving path (Default);
            nonstandard residues; missing residues;
            missing atoms; missing terminals; removed heterogen names;
            removed one-length chains; MET with Se residue id; etc;
    """
    from pdbfixer import PDBFixer
    try:
        import openmm
        from openmm import app
    except ImportError:  # OpenMM < 7.6
        from simtk import openmm
        from simtk.openmm import app

    assert isinstance(renumber_Residues, bool), "args `renumber_Residues` Bool type needed!"
    assert isinstance(addHs_pH, (int, float)), "args `addHs_pH` int or float type needed!"
    info_dict = dict(output = output)

    fixpro = PDBFixer(
        filename = filename,
        pdbfile = None,
        pdbxfile = None,
        pdbid = None,
        url = None
    )
    info_fn = partial(
        _get_info_if_nonnone, dic = info_dict,
        pdbfixer = fixpro, return_infol = return_info)
    if replace_nonstandard_aa:
        fixpro.findNonstandardResidues()
        info_fn('nonstandardResidues')
        fixpro.replaceNonstandardResidues()
    if remove_heterogens:
        _remove_heterogens(fixpro, info_dict, keep_water = True)
    if add_missing:
        fixpro.findMissingResidues()
        info_fn('missingResidues')
        fixpro.findMissingAtoms()
        info_fn('missingAtoms')
        info_fn('missingTerminals')
        fixpro.addMissingAtoms()
    if hydrogenate:
        fixpro.addMissingHydrogens(pH = float(addHs_pH))
    app.PDBFile.writeFile(
        topology = fixpro.topology,
        positions = fixpro.positions,
        file = open(output, 'w'),
        keepIds = True,
    )

    if renumber_Residues:
        try:
            original = mda.Universe(filename)
            from_fixpro = mda.Universe(output)

            resNum = [res.resid for res in original.residues]
            for idx, res in enumerate(from_fixpro.residues):
                res.resid = resNum[idx]

            save = PDB.PDBWriter(filename = output)
            save.write(from_fixpro)
            save.close()
        except Exception as e:
            print('Can not renumber residues. Please check exception for extra details.\n'
                  f'Exception message is:\n{e}')
    return info_dict

def _get_info_if_nonnone(
        attr_name: str,
        dic: Dict[str, Any],
        pdbfixer,
        return_info: bool = True,
):
    if not return_info:
        return
    dic[attr_name] = getattr(pdbfixer, attr_name)

def _remove_heterogens(
        fixer,
        alterations_info: dict,
        keep_water: bool = True,
):
    """
    Removes the residues that Pdbfixer considers to be heterogens.
    Args:
      fixer: A Pdbfixer instance.
      alterations_info: A dict that will store details of changes made.
      keep_water: If True, water (HOH) is not considered to be a heterogen.
    """
    initial_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            initial_resnames.add(residue.name)
    fixer.removeHeterogens(keepWater = keep_water)
    final_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            final_resnames.add(residue.name)
    alterations_info["removed_heterogens"] = initial_resnames.difference(
        final_resnames
    )
