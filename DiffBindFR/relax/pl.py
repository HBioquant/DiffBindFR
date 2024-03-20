# Copyright (c) MDLDrugLib. All rights reserved.
import os
import shutil
import typing as t
import argparse
import subprocess
import pandas as pd
import warnings
import logging
from pathlib import Path
import multiprocessing as mp
import numpy as np
import mdtraj
import openmm as mm
import openmm.app as mm_app
import openmm.unit as mm_unit
from openmm import CustomExternalForce
import pdbfixer
from openmm.app import Modeller
import openff
from openff.toolkit import Molecule
from openff.units import Quantity as openff_Quantity
from openff.units.openmm import from_openmm, to_openmm
from openff.toolkit.utils.exceptions import OpenFFToolkitException
from openmmforcefields.generators import SystemGenerator

from rdkit import Chem

from pandarallel import pandarallel

from Bio.PDB import (
    PDBParser, MMCIFParser,
    PDBIO, Select, MMCIFIO,
)
logger = logging.getLogger('PLRelaxer')
warnings.filterwarnings("ignore")

def fix_pdb(
        input_pdb_file: str,
        out_fixed_pdb_file: t.Optional[str] = None,
        keepIds: bool = False,
        seed: t.Optional[int] = None,
):
    """Preprocessing of protein."""
    if not Path(input_pdb_file).exists():
        raise FileNotFoundError(input_pdb_file)
    fixer = pdbfixer.PDBFixer(filename=input_pdb_file)
    fixer.removeHeterogens()
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=seed)
    fixer.addMissingHydrogens(7.0)
    if out_fixed_pdb_file is not None:
        Path(out_fixed_pdb_file).parent.mkdir(parents=True, exist_ok=True)
        mm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open(str(out_fixed_pdb_file), 'w'), keepIds=keepIds)
    return fixer.topology, fixer.positions

def load_protein(
        protein_file: str,
):
    if not Path(protein_file).exists():
        raise FileNotFoundError(protein_file)
    protein_file = str(protein_file)
    if protein_file.endswith('.pdb'):
        protein = mm_app.PDBFile(protein_file)
    elif protein_file.endswith('.cif'):
        protein = mm_app.PDBxFile(protein_file)
    else:
        suffix = Path(protein_file).suffix
        raise NotImplementedError(f".pdb or .cif are supported, but {suffix} from {protein_file}")

    return protein

def remove_hydrogen_pdb(pdbFile, toFile):

    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    class NoHydrogen(Select):
        def accept_atom(self, atom):
            if atom.element == 'H' or atom.element == 'D':
                return False
            return True

    io = MMCIFIO() if toFile[-4:] == ".cif" else PDBIO()
    io.set_structure(s)
    io.save(toFile, select=NoHydrogen())

def unicon(
    molf,
    outputf,
    exec_bin = '/data02/zhujintao/tools/zbh_tools/unicon_1.4.2/unicon',
    **kwargs
):
    work_dir = os.path.dirname(molf)
    basename = os.path.basename(molf)
    params = f'cd {work_dir} && {exec_bin} -i {basename} -o {outputf}'
    subprocess.call(params, shell = True, **kwargs)
    return

def parse_molfile(
    path: Path,
    sanitize=True,
    removeHs=True,
    strictParsing=True,
    proximityBonding=True,
    cleanupSubstructures=True,
    allow_rescue=True,
    **kwargs,
) -> Chem.rdchem.Mol:
    """Load one molecule from a file, picking the right RDKit function."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".sdf":
        mol = Chem.MolFromMolFile(
            str(path),
            sanitize=sanitize,
            removeHs=removeHs,
            strictParsing=strictParsing,
        )
        if mol is None and allow_rescue:
            unicon(
                str(path),
                Path(path).stem + '.mol2',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _mol_path = Path(path).parent / (Path(path).stem + '.mol2')
            mol = parse_molfile(
                _mol_path,
                sanitize=sanitize,
                removeHs=removeHs,
                strictParsing=strictParsing,
                proximityBonding=proximityBonding,
                cleanupSubstructures=cleanupSubstructures,
                allow_rescue=False,
                **kwargs,
            )
    elif path.suffix == ".mol2":
        mol = Chem.MolFromMol2File(
            str(path),
            sanitize=sanitize,
            removeHs=removeHs,
            cleanupSubstructures=cleanupSubstructures,
        )
        if mol is None and allow_rescue:
            unicon(
                str(path),
                Path(path).stem + '.sdf',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _mol_path = Path(path).parent / (Path(path).stem + '.sdf')
            if _mol_path.exists() and os.path.getsize(_mol_path) != 0:
                mol = parse_molfile(
                    _mol_path,
                    sanitize=sanitize,
                    removeHs=removeHs,
                    strictParsing=strictParsing,
                    proximityBonding=proximityBonding,
                    cleanupSubstructures=cleanupSubstructures,
                    allow_rescue=False,
                    **kwargs,
                )
    elif path.suffix == ".pdb":
        mol = Chem.MolFromPDBFile(
            str(path),
            sanitize=sanitize,
            removeHs=removeHs,
            proximityBonding=proximityBonding,
        )
        if mol is None and proximityBonding:
            mol = Chem.MolFromPDBFile(
                str(path),
                sanitize=sanitize,
                removeHs=removeHs,
                proximityBonding=False,
            )
        if mol is None and allow_rescue:
            unicon(
                str(path),
                Path(path).stem + '.sdf',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _mol_path = Path(path).parent / (Path(path).stem + '.sdf')
            if _mol_path.exists() and os.path.getsize(_mol_path) != 0:
                mol = parse_molfile(
                    _mol_path,
                    sanitize=sanitize,
                    removeHs=removeHs,
                    strictParsing=strictParsing,
                    proximityBonding=proximityBonding,
                    cleanupSubstructures=cleanupSubstructures,
                    allow_rescue=False,
                    **kwargs,
                )
    elif path.suffix == ".mol":
        block = "".join(open(path).readlines()).strip() + "\nM  END"
        mol = Chem.MolFromMolBlock(
            block,
            sanitize=sanitize,
            removeHs=removeHs,
            strictParsing=strictParsing,
        )
    else:
        raise ValueError(f"Unknown file type {path.suffix}")

    if mol is not None:
        mol.SetProp("_Path", str(path))

    return mol

def load_mol(
        ligand_file: str,
        **kwargs,
):
    ligand_file = Path(ligand_file)
    mol = parse_molfile(
        path = ligand_file,
        sanitize = kwargs.get('sanitize', True),
        removeHs = kwargs.get('removeHs', True),
        strictParsing = kwargs.get('strictParsing', True),
        proximityBonding = kwargs.get('proximityBonding', True),
        cleanupSubstructures = kwargs.get('cleanupSubstructures', True),
    )
    if mol is None:
        return None, ''
    if mol.GetNumConformers() < 0.5:
        raise ValueError(f'mol from {ligand_file} has no conformer.')
    mol_h = Chem.AddHs(mol, addCoords=True)
    smiles = Chem.MolToSmiles(mol_h)
    m_order = list(
        mol_h.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol_h = Chem.RenumberAtoms(mol_h, m_order)

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.add_conformer(mm_unit.Quantity(mol_h.GetConformer().GetPositions(), mm_unit.angstrom))

    try:
        molecule.assign_partial_charges(partial_charge_method='gasteiger')
    except Exception as e:
        logger.warning(f'partial charge failed for mol from {ligand_file}. Set Zeros.\nERROR: {str(e)}')
        molecule.assign_partial_charges(partial_charge_method='zeros')

    return molecule, smiles

def remove_hydrogen_reorder(mol):
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol)
    m_order = list(
        mol.GetPropsAsDict(
            includePrivate=True,
            includeComputed=True,
        )["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, m_order)
    return mol

def define_restrain(
        atom: mm_app.Atom,
        rst_type: str = "CA",
):
    if rst_type == "non_H":
        return atom.element.name != "hydrogen"
    elif rst_type == "CA":
        return atom.name == "CA"
    elif rst_type == 'protein':
        return 'x' not in atom.name
    elif rst_type == "CA+ligand":
        return ('x' in atom.name) or (atom.name == "CA")
    elif rst_type == "ligand":
        return 'x' in atom.name
    else:
        raise NotImplementedError(rst_type)

def set_pr_system(topology):
    """
    Set the system using the topology from the pdb file
        for protein_only relaxation.
    """
    #Put it in a force field to skip adding all particles manually
    forcefield = mm_app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(topology,
                                     removeCMMotion=False,
                                     nonbondedMethod=mm_app.NoCutoff,
                                     rigidWater=True #Use implicit solvent
                                     )
    return system

def add_p_restraints(
        system,
        topology,
        positions,
        n_res: int,
        restraint_type: str,
        restraint_mask: t.Optional[np.ndarray] = None,
        stiffness: float = 500.0,
):
    restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.addGlobalParameter('k', stiffness * mm_unit.kilojoules_per_mole/mm_unit.nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atid, atom in enumerate(topology.atoms()):
        if atom.residue.index >= n_res:
            continue

        if restraint_mask is not None and restraint_mask[atid]:
            continue

        if define_restrain(atom, restraint_type):
            restraint.addParticle(atom.index, positions[atom.index])

    system.addForce(restraint)
    return system

def add_l_restraints(
        system,
        topology,
        positions,
        n_res: int,
        restraint_type: str,
        stiffness: float = 1000.0,
):
    restraint = CustomExternalForce('k_ligand*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.addGlobalParameter('k_ligand', stiffness * mm_unit.kilojoules_per_mole/mm_unit.nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atid, atom in enumerate(topology.atoms()):
        if atom.residue.index < n_res:
            continue

        if define_restrain(atom, restraint_type):
            restraint.addParticle(atom.index, positions[atom.index])

    system.addForce(restraint)
    return system

def minimize_energy(
        topology,
        system,
        positions,
        reportInterval = 0,
        gpu: bool = True,
        tolerance = 0.01, # as posebusters do 0.01 kj/mol (redock? cross dock with sc packing should be higher)
        maxIterations = 0, # minimization is continued until the results converge without regard
):
    """Function that minimizes energy, given topology, OpenMM system, and positions"""
    # integrator = mm.LangevinIntegrator(0, 0.01, 0.0)
    integrator = mm.LangevinIntegrator(300 * mm_unit.kelvin, 1 / mm_unit.picosecond, 0.002 * mm_unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA" if gpu else "CPU")
    simulation = mm.app.Simulation(topology, system, integrator, platform)

    if reportInterval > 0:
        # Initialize the DCDReporter
        reporter = mdtraj.reporters.DCDReporter('traj.dcd', reportInterval)
        # Add the reporter to the simulation
        simulation.reporters.append(reporter)

    simulation.context.setPositions(positions)

    ENERGY = mm_unit.kilocalories_per_mole
    LENGTH = mm_unit.angstroms
    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

    try:
        simulation.minimizeEnergy(
            tolerance * mm_unit.kilojoule_per_mole / mm_unit.nanometer,
            maxIterations)
    except Exception as e:
        logger.info(f'Error When energy minimization: {str(e)}')
        # openmm.OpenMMException: Particle coordinate is nan in EDM-Dock
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        if reportInterval > 0:
            reporter.close()
        return ret
    # Save positions
    minstate = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = minstate.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = minstate.getPositions(asNumpy=True).value_in_unit(LENGTH)

    if reportInterval > 0:
        reporter.close()

    logger.info(f'Energy change: {ret["einit"]} -> {ret["efinal"]}')

    return ret

def export_openff_mol(
        openff_mol,
        to_file,
):
    new_mol = openff_mol.to_rdkit()
    new_mol = remove_hydrogen_reorder(new_mol)
    Path(to_file).parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(to_file))
    w.write(new_mol)
    w.close()
    return

def export_obj(
        min_ret,
        modeller,
        protein_topology,
        smiles: t.Optional[str] = None,
        out_relax_pdb_file: t.Optional[str] = None,
        out_relax_lig_file: t.Optional[str] = None,
        out_relax_complex_file: t.Optional[str] = None,
        **kwargs,
):
    if out_relax_pdb_file is not None:
        if out_relax_pdb_file.endswith('.pdb'):
            export_fn = mm_app.PDBFile.writeFile
        elif out_relax_pdb_file[-3:] == 'cif':
            export_fn = mm_app.PDBxFile.writeFile
        else:
            suffix = Path(out_relax_pdb_file).suffix
            raise NotImplementedError(f"{suffix} from {out_relax_pdb_file} not supported (.pdb or .cif)")
        Path(out_relax_pdb_file).parent.mkdir(parents=True, exist_ok=True)
        export_fn(
            protein_topology,
            min_ret["pos"][:protein_topology.getNumAtoms()],
            open(out_relax_pdb_file, 'w'),
            keepIds=kwargs.get('keepIds', True),
        )
        remove_hydrogen_pdb(out_relax_pdb_file, out_relax_pdb_file)
        if out_relax_complex_file is not None:
            Path(out_relax_complex_file).parent.mkdir(parents=True, exist_ok=True)
            export_fn(
                modeller.topology,
                min_ret["pos"],
                open(out_relax_complex_file, 'w'),
                keepIds=kwargs.get('keepIds', True),
            )

    if out_relax_lig_file is not None:
        new_molecule = Molecule.from_smiles(
            smiles, allow_undefined_stereo=True,
        )
        new_molecule.add_conformer(
            mm_unit.Quantity(
                min_ret["pos"][protein_topology.getNumAtoms():],
                mm_unit.angstrom)
        )
        export_openff_mol(new_molecule, out_relax_lig_file)

    return

def relax_pl(
        input_pdb_file: str,
        ligand_file: t.Optional[str] = None,
        out_fixed_pdb_file: t.Optional[str] = None,
        out_relax_pdb_file: t.Optional[str] = None,
        out_relax_lig_file: t.Optional[str] = None,
        out_relax_complex_file: t.Optional[str] = None,
        **kwargs,
):
    ## Skip already done system
    xs = [out_relax_pdb_file, out_relax_lig_file, out_relax_complex_file]
    if all(x is None for x in xs):
        return
    if all(Path(x).exists() for x in xs if x is not None):
        return

    ## Read protein PDB and add hydrogens
    logger.info('Preprocessing protein using pdbfixer...')
    protein_topology, protein_positions = fix_pdb(
        input_pdb_file = input_pdb_file,
        out_fixed_pdb_file = out_fixed_pdb_file,
        keepIds = kwargs.get('keepIds', True),
        seed = kwargs.get('seed', None),
    )
    n_res = protein_topology.getNumResidues()

    logger.info('Preparing protein-ligand complex...')
    modeller = Modeller(protein_topology, protein_positions)
    logger.info('Protein-Only System has %d atoms of %d residues' % (
        modeller.topology.getNumAtoms(), n_res,
    ))

    ligand_mol, smiles = None, None
    if ligand_file is not None:
        ligand_mol, smiles = load_mol(
            ligand_file = ligand_file,
            sanitize=kwargs.get('sanitize', True),
            removeHs=kwargs.get('removeHs', True),
            strictParsing=kwargs.get('strictParsing', True),
            proximityBonding=kwargs.get('proximityBonding', True),
            cleanupSubstructures=kwargs.get('cleanupSubstructures', True),
        )
        if ligand_mol is None:
            logger.error(f'Error in parsing ligand file: {ligand_file}')
            if out_relax_lig_file is not None:
                shutil.copy(ligand_file, out_relax_lig_file)
            if out_relax_pdb_file:
                if out_relax_pdb_file.endswith('.pdb'):
                    export_fn = mm_app.PDBFile.writeFile
                elif out_relax_pdb_file[-3:] == 'cif':
                    export_fn = mm_app.PDBxFile.writeFile
                else:
                    suffix = Path(out_relax_pdb_file).suffix
                    raise NotImplementedError(f"{suffix} from {out_relax_pdb_file} not supported (.pdb or .cif)")
                Path(out_relax_pdb_file).parent.mkdir(parents=True, exist_ok=True)
                export_fn(
                    protein_topology,
                    protein_positions,
                    open(out_relax_pdb_file, 'w'),
                    keepIds=kwargs.get('keepIds', True),
                )
            # no complex file output to indicate this is placeholder and no minimization performed.
            return

        lig_top = ligand_mol.to_topology()
        modeller.add(lig_top.to_openmm(), ligand_mol.conformers[0])
        logger.info('Complex System has %d atoms' % modeller.topology.getNumAtoms())

    logger.info('Prepare system...')
    forcefield_kwargs = {'constraints': mm_app.HBonds, }

    try:
        # Load the ff14sb and OpenFF "Sage" force field.
        system_generator = SystemGenerator(
            forcefields=['amber/protein.ff14SB.xml'],
            small_molecule_forcefield='openff-2.0.0',
            molecules=[ligand_mol] if ligand_mol is not None else [],
            forcefield_kwargs=forcefield_kwargs,
        )
        if ligand_mol is not None:
            system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
        else:
            system = system_generator.create_system(modeller.topology)
    except:
        try:
            # if sage ff fail to parameterize the ligand, use gaff to rescue
            system_generator = SystemGenerator(
                forcefields=['amber/protein.ff14SB.xml'],
                small_molecule_forcefield='gaff-2.11',
                molecules=[ligand_mol] if ligand_mol is not None else [],
                forcefield_kwargs=forcefield_kwargs,
            )
            if ligand_mol is not None:
                system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
            else:
                system = system_generator.create_system(modeller.topology)
        except Exception as e:
            logger.error(f'Error when build system: {str(e)}')
            # raise No template found for residues
            # exceptions.UnassignedProperTorsionParameterException: ProperTorsionHandler was not able to find parameters for the following valence terms
            # make output placeholder
            if out_relax_lig_file is not None:
                export_openff_mol(ligand_mol, out_relax_lig_file)
            if out_relax_pdb_file:
                if out_relax_pdb_file.endswith('.pdb'):
                    export_fn = mm_app.PDBFile.writeFile
                elif out_relax_pdb_file[-3:] == 'cif':
                    export_fn = mm_app.PDBxFile.writeFile
                else:
                    suffix = Path(out_relax_pdb_file).suffix
                    raise NotImplementedError(f"{suffix} from {out_relax_pdb_file} not supported (.pdb or .cif)")
                Path(out_relax_pdb_file).parent.mkdir(parents=True, exist_ok=True)
                export_fn(
                    protein_topology,
                    protein_positions,
                    open(out_relax_pdb_file, 'w'),
                    keepIds=kwargs.get('keepIds', True),
                )
            # no complex file output to indicate this is placeholder and no minimization performed.
            return

    # give some residues (such as pocket to relex)
    rst_mask = kwargs.get('rst_mask', None)
    if rst_mask is not None:
        rst_mask = rst_mask.astype('bool')
        assert rst_mask.shape[0] == n_res, \
            f'inconsistent residue number input protein {n_res}' \
            f' vs rst mask {rst_mask.shape[0]}'

    logger.info('Add restraint...')
    system = add_p_restraints(
        system,
        modeller.topology,
        modeller.positions,
        n_res=n_res,
        restraint_type=kwargs.get("p_restraint_type", 'protein'),
        restraint_mask=rst_mask,
        stiffness=float(kwargs.get('p_stiffness', 500.0))
    )
    l_stiffness = float(kwargs.get('l_stiffness', 1000.0))
    if ligand_mol is not None and l_stiffness > 0.:
        system = add_l_restraints(
            system,
            modeller.topology,
            modeller.positions,
            n_res=n_res,
            restraint_type=kwargs.get("l_restraint_type", 'non_H'),
            stiffness=l_stiffness,
        )

    ## Minimize energy
    logger.debug('Running Minimization...')
    ret = minimize_energy(
        modeller.topology,
        system,
        modeller.positions,
        gpu=kwargs.get('gpu', True),
        reportInterval=kwargs.get('ccd_int', 0),
        tolerance=kwargs.get('tolerance', 1),
        maxIterations=kwargs.get('maxIterations', 0),
    )

    export_obj(
        ret,
        modeller,
        protein_topology,
        smiles,
        out_relax_pdb_file,
        out_relax_lig_file if ligand_mol is not None else None,
        out_relax_complex_file if ligand_mol is not None else None,
        **kwargs,
    )

    if ligand_mol is not None:
        logger.debug(f'Complex: {input_pdb_file} with ligand {ligand_file} minimization is done!')
    else:
        logger.debug(f'Protein: {input_pdb_file} minimization is done!')

    return

def minimizer(
        work_dir,
        relax_protein_first = True,
        **kwargs,
):
    pandarallel.initialize(
        nb_workers=kwargs.get("num_workers", 1),
        progress_bar=kwargs.get("verbose", False),
    )
    tasks = os.listdir(work_dir)
    _tasks = []
    for CID in tasks:
        all_done = []
        for samples in os.listdir(os.path.join(work_dir, CID)):
            DIR = os.path.join(work_dir, CID, samples)
            out_relax_pdb_file = f'{DIR}/relaxed_protein.pdb'  # None
            out_relax_lig_file = f'{DIR}/relaxed_ligand.sdf'  # None
            if os.path.exists(out_relax_lig_file) and os.path.exists(out_relax_pdb_file):
                all_done.append(True)
            else:
                all_done.append(False)
        if not all(all_done):
            _tasks.append(CID)
    tasks = _tasks
    logger.info(f'Number of tasks: ', len(tasks))
    df = pd.DataFrame({'task': tasks})

    def process(row):
        CID = row['task']
        for samples in os.listdir(os.path.join(work_dir, CID)):
            DIR = os.path.join(work_dir, CID, samples)
            input_pdb_file = f'{DIR}/prot_final.pdb'
            ligand_file = f'{DIR}/lig_final_ec.sdf'
            if os.path.exists(ligand_file):
                ligand_file = f'{DIR}/lig_final.sdf'

            out_fixed_pdb_file = f'{DIR}/fixed.pdb'
            out_relax_pdb_file = f'{DIR}/relaxed_protein.pdb'
            out_relax_lig_file = f'{DIR}/relaxed_ligand.sdf'
            out_relax_complex_file = f'{DIR}/relaxed_complex.pdb'

            if relax_protein_first:
                relax_pl(
                    input_pdb_file,
                    None,
                    out_fixed_pdb_file,
                    out_relax_pdb_file,
                    None,
                    None,
                    **kwargs,
                )
                input_pdb_file = out_relax_pdb_file

            relax_pl(
                input_pdb_file,
                ligand_file,
                out_fixed_pdb_file,
                out_relax_pdb_file,
                out_relax_lig_file,
                out_relax_complex_file,
                **kwargs,
            )
        return

    if kwargs.get("num_workers", 1) > 1:
        df.parallel_apply(process, axis=1)
    else:
        df.apply(process, axis=1)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein-ligand complex system MM-based relaxation.')
    parser.add_argument(
        'work_dir', type = str,
        help = 'Binding Complex directory with the tree [work_dir]/[ComplexID]/[EnsembleID].'
               'Every ensemble directory has lig_final.sdf (or lig_final_ec.sdf) and prot_final.pdb',
    )
    parser.add_argument(
        '-nb',
        '--num_workers',
        type=int,
        default=mp.cpu_count() // 2,
        help='The number of workers for multi-processing. '
             'Defaults to the half number of available cpus.'
    )
    parser.add_argument(
        '-cpu',
        '--use_cpu',
        action='store_true',
        default = False,
        help='Run OpenMM on CPU device rather than GPU acceleration.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Whether show the progress bar.'
    )
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG)
    kwargs = dict(
        sanitize=True,
        removeHs=True,
        strictParsing=True,
        proximityBonding=True,
        cleanupSubstructures=True,
        p_restraint_type='protein',
        p_stiffness=100.,
        l_restraint_type='non_H',
        l_stiffness=0.,
        tolerance=0.01,
        maxIterations=0,
        gpu=(not args.use_cpu),
        ccd_int=0,
        keepIds=True,
        seed=None,
        rst_mask=None,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
    minimizer(
        work_dir = args.work_dir,
        **kwargs
    )
