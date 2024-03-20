# Copyright (c) MDLDrugLib. All rights reserved.
import random, os
from typing import Tuple
from rdkit import Chem
try:
    import py3Dmol
except ImportError:
    py3Dmol = None


def get_3dview(
        receptor:str,
        docking_results:str,
        refMol:str,
        rep_opts:dict = {'format' : 'pdb'},
        refMol_opts:dict = {'format' : 'mol2'},
        pose:list = [0]
) -> None:
    view = py3Dmol.view()
    view.removeAllModels()
    view.setViewStyle({
        'style' : 'outline',
        'color' : 'black',
        'width' : 0.1,
    })
    view.addModel(open(receptor, 'r').read(), **rep_opts)
    protein = view.getModel()
    protein.setStyle({
        'cartoon' : {
            'arrow' : True,
            'tubes' : True,
            'style' : 'oval',
            'color' : 'white',
        }
    })

    if refMol:
        view.addModel(open(refMol, 'r').read(), **refMol_opts)
        refm = view.getModel()
        refm.setStyle(
            {}, {
                'stick' : {
                    'colorscheme' : 'greenCarbon',
                    'radius' : 0.2
                }
            }
        )
    if pose:
        results = Chem.SDMolSupplier(docking_results)
        for index in pose:
            color = ["#" + "".join([random.choice('0123456789ABCDEF') for j in range(6)])]
            p = Chem.MolToMolBlock(results[index])
            view.addModel(p, 'mol')
            x = view.getModel()
            x.setStyle(
                {}, {
                    'color' : color[0],
                    'radius' : 0.1,
                }
            )
    view.zoomTo()
    view.show()

def visualize_protein_ligand(
        pdb_block: str,
        sdf_block: str,
        show_ligand: bool = True,
        show_surface: bool = True,
) -> py3Dmol.view:
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.setStyle(
        {'model': -1},
        {'cartoon': {'color': 'spectrum'},
         'line': {}},
    )

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
        if show_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view

def visualize_data(
        data,
        root,
        show_ligand = True,
        show_surface = True,
) -> py3Dmol.view:
    protein_path = os.path.join(root, data.protein_file)
    ligand_path = os.path.join(root, data.ligand_file)
    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()
    return visualize_protein_ligand(
        pdb_block, sdf_block,
        show_ligand = show_ligand,
        show_surface = show_surface)

def visualize_generated_mol(
        protein_file: str,
        mol: Chem.rdchem.Mol,
        show_surface: bool = False,
        opacity: float = 0.5,
) -> py3Dmol.view:
    with open(protein_file, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})

    mblock = Chem.MolToMolBlock(mol)
    view.addModel(mblock, 'mol')
    view.setStyle({'model': -1}, {'stick': {}, 'sphere': {'radius': 0.35}})
    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': opacity}, {'model': -1})

    view.zoomTo()
    return view


def MolTo3DView(
        mol: Chem.rdchem.Mol,
        size: Tuple[float] = (300, 300),
        style: str = "stick",
        surface: bool = False,
        opacity: float = 0.5,
):
    """
    Draw molecule in 3D
    Args:
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')

    viewer = py3Dmol.view(width = size[0], height = size[1])
    if isinstance(mol, list):
        for i, m in enumerate(mol):
            mblock = Chem.MolToMolBlock(m)
            viewer.addModel(mblock, 'mol' + str(i))
    elif len(mol.GetConformers()) > 1:
        for i in range(len(mol.GetConformers())):
            mblock = Chem.MolToMolBlock(mol, confId = i)
            viewer.addModel(mblock, 'mol' + str(i))
    else:
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})

    viewer.zoomTo()
    return viewer