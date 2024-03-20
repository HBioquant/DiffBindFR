# Copyright (c) MDLDrugLib. All rights reserved.
import os
import re
import sys
import logging
import argparse
import os.path as osp
from typing import List
from collections import defaultdict

import numpy as np
import prody
prody.LOGGER._logger.setLevel(logging.INFO)
from prody import parsePDB, writePDB, writePDBStream


def split_top1_flex_pdbqt(
        docked_pdbqt: str,
) -> str:
    if not osp.exists(docked_pdbqt):
        raise FileNotFoundError(docked_pdbqt)

    with open(docked_pdbqt, 'r') as f:
        lines = f.readlines()

    top1 = []
    is_top1, is_sc = False, False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('ENDMDL'): break

        if line.startswith('MODEL 1'):
            is_top1 = True

        if is_top1 and line.startswith('BEGIN_RES'):
            is_sc = True

        if is_sc:
            top1.append(line)

    return '\n'.join(top1)

def parse_top_flex_pdbqt(
        flex_pdbqt_lines: List[str],
) -> defaultdict:
    """
    Build a residues-atoms mapping with 3D coordinates.
    Args:
        flex_pdbqt_lines: the top1 flexible residues pdbqt context
            separated by vinasplit or by func:split_top1_flex_pdbqt:
    """
    mapping = defaultdict(dict)

    if not flex_pdbqt_lines:
        raise ValueError("Empty file.")

    line_count = 0
    current_res = None
    for line in flex_pdbqt_lines:
        line_count += 1
        if not line.strip():
            continue  # skip empty lines
        if line.startswith('BEGIN_RES'):
            x = line.split()[1:]
            # such as BEGIN_RES LEU A1035
            if len(x) == 2:
                pattern = re.compile(r'\d+')
                resnum = re.findall(pattern, x[-1])[0]
                chain = x[-1].replace(resnum, '')
                x = [x[0], chain, resnum]
            elif len(x) == 1:
                raise NotImplementedError(line)
            current_res = f'{x[1]}:{x[2]}:{x[0]}'
            # print(current_res)
        elif line.startswith('ATOM'):
            fullname = line[12:16]
            # get rid of whitespace in atom names
            split_list = fullname.split()
            if len(split_list) != 1:
                # atom name has internal spaces, e.g. " N B ", so
                # we do not strip spaces
                name = fullname
            else:
                # atom name is like " CA ", so we can strip spaces
                name = split_list[0]

            altloc = line[16]
            resname = line[17:20].strip()
            chainid = line[21]
            resseq = int(line[22:26].split()[0])  # sequence identifier

            # atomic coordinates
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                raise ValueError(
                    f"Invalid or missing coordinate(s) at line {line_count}."
                )

            current_expr = f'{chainid}:{resseq}:{resname}'
            if current_expr != current_res:
                raise ValueError(
                    f"residue mismatch at line {line_count} when current atom residue is {current_expr} while the current residue should be {current_res}."
                )

            # print(f'{current_res} >> {name}{altloc} {resname} {chainid} {resseq} {x} {y} {z}')
            mapping[current_res][name] = (x, y, z)

    return mapping

def select_vinafr_res(
        protein: prody.AtomGroup,
        res_expr: str, # 'chain:residue number:residue name'
) -> prody.Residue:
    res_expr = res_expr.split(':')
    res_expr = f'chain {res_expr[0]} and resnum {res_expr[1]} and resname {res_expr[2]}'

    residue = protein.select(f'{res_expr} and not element H and not name C O N')
    if residue is None:
        raise ValueError(f'{res_expr} does not exist.')

    return residue

def get_res_hydrogen_expr(
        residues_exprs: List[str],
) -> str:
    new_expr = []
    for expr in residues_exprs:
        chainid, resnum, resname = expr.split(':')
        new_expr.append(
            f'(chain {chainid} and resnum {resnum} and resname {resname} and element H)'
        )
    return ' or '.join(new_expr)

def remodelling(
        reference_pdb: str,
        flex_pdbqt_lines: List[str],
) -> prody.AtomGroup:
    res_atm_coords = parse_top_flex_pdbqt(flex_pdbqt_lines)
    protein = parsePDB(reference_pdb)

    for res in res_atm_coords.keys():
        residue = select_vinafr_res(
            protein, res,
        )
        for at in residue.iterAtoms():
            atn = at.getName()
            new_atoms = res_atm_coords[res]
            if atn not in new_atoms:
                raise ValueError(
                    f'Atom {atn} from {reference_pdb} does not exist in {res}'
                )

            new_coord = new_atoms[atn]
            at.setCoords(np.array(new_coord))

    # Remove Hs as the heavy atom coordinates has been removed so the orignal H is invalid
    flex_res = list(res_atm_coords.keys())
    hydrogen_expr = get_res_hydrogen_expr(flex_res)
    protein = protein.select(f'not ({hydrogen_expr})')

    return protein

def build_vinafr_protein(args: argparse.Namespace) -> None:

    if not osp.exists(args.flex_pdbqt):
        raise FileNotFoundError(args.flex_pdbqt)

    with open(args.flex_pdbqt, 'r') as f:
        lines = f.read()

    if 'MODEL 1' in lines:
        lines = split_top1_flex_pdbqt(args.flex_pdbqt)
    lines = lines.split('\n')

    updated_protein = remodelling(
        args.reference, lines,
    )

    if args.output is not None:
        os.makedirs(osp.dirname(args.output), exist_ok=True)
        writePDB(args.output, updated_protein)
    else:
        writePDBStream(sys.stdout, updated_protein)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=str,
                        help="Vinafr input protein PDB file.")
    parser.add_argument("flex_pdbqt", type=str,
                        help="flex pdbqt.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save updated protein to the file.")
    args = parser.parse_args()
    build_vinafr_protein(args)