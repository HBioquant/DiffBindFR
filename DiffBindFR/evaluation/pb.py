# Copyright (c) MDLDrugLib. All rights reserved.
from __future__ import annotations
import sys
import copy
import logging
import argparse
import inspect
import signal
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import (
    Any, Iterable, Callable, Generator,
)

import pandas as pd
from yaml import safe_load
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromPDBBlock

try:
    import posebusters
except ImportError as e:
    print('Package posebusters has not been installed. Please `pip install posebusters`')
    sys.exit(1)
from posebusters.posebusters import _dataframe_from_output
from posebusters.tools.formatting import create_long_output, create_short_output

from posebusters.tools.loading import (
    CaptureLogger, _check_path, _load_mol,
    safe_supply_mols, _process_mol,
)
from posebusters.modules.distance_geometry import check_geometry
from posebusters.modules.energy_ratio import check_energy_ratio
from posebusters.modules.flatness import check_flatness
from posebusters.modules.identity import check_identity
from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from posebusters.modules.loading import check_loading
from posebusters.modules.rmsd import check_rmsd
from posebusters.modules.sanity import check_chemistry
from posebusters.modules.volume_overlap import check_volume_overlap
from posebusters.posebusters import PoseBusters

logger = logging.getLogger(__name__)

@contextmanager
def time_limit(seconds, *args, **kwargs):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out during {seconds}s.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def _safe_supply_mols(*args, **kwargs):
    with time_limit(60):
        return safe_supply_mols(*args, **kwargs)


def _format_results(df: pd.DataFrame, outfmt: str = "short", no_header: bool = False, index: int = 0) -> str:
    if outfmt == "long":
        return create_long_output(df)
    elif outfmt == "csv":
        header = (not no_header) and (index == 0)
        df.index.names = ["file", "molecule"]
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df.to_csv(index=True, header=header)
    elif outfmt == "short":
        return create_short_output(df)
    else:
        raise ValueError(f"Unknown output format {outfmt}")


def _select_mode(config, columns: Iterable[str]) -> str | dict[str, Any]:
    # decide on mode to run

    # load config if provided
    if type(config) == Path:
        return dict(safe_load(open(config)))

    # forward string if config provide
    if type(config) == str:
        return str(config)

    # select mode based on inputs
    if "mol_pred" in columns and "mol_true" in columns and "mol_cond" in columns:
        mode = "redock"
    elif "mol_pred" in columns and ("protein" in columns) or ("mol_cond" in columns):
        mode = "dock"
    elif any(column in columns for column in ("mol_pred", "mols_pred", "molecule", "molecules", "molecule")):
        mode = "mol"
    else:
        raise NotImplementedError(f"No supported columns found in csv. Columns found are {columns}")

    return mode

all_module_dict: dict[str, Callable] = {
    "loading": check_loading,
    "sanity": check_chemistry,
    "identity": check_identity,
    "distance_geometry": check_geometry,
    "flatness": check_flatness,
    "energy_ratio": check_energy_ratio,
    "intermolecular_distance": check_intermolecular_distance,
    "volume_overlap": check_volume_overlap,
    "rmsd": check_rmsd,
}

def load_mol(
        path: Path | str | Mol,
        *args,
        **kwargs,
):
    if isinstance(path, Mol):
        return path
    else:
        return _load_mol(path, *args, **kwargs)

def safe_load_mol(
        path: Path | str | Mol,
        load_all: bool = False,
        **load_params,
) -> Mol | None:
    """Load one molecule from a file, optionally adding hydrogens and assigning bond orders.

    Args:
        path: Path to file containing molecule.

    Returns:
        Molecule object or None if loading failed.
    """
    try:
        with time_limit(60):
            if not isinstance(path, Mol):
                path = _check_path(path)
            with CaptureLogger():
                mol = load_mol(path, load_all=load_all, **load_params)
            return mol
    except Exception as exception:
        logger.warning(f"Could not load molecule from {path} with error: {exception}")
    return None

class PoseChecker(PoseBusters):
    def __init__(
            self,
            config: str | dict[str, Any] = "redock",
            top_n: int | None = None,
            module_dict: dict[str, Callable] | None = None,
            subset: Iterable[str] | str | None = None,
    ):
        super().__init__(config, top_n)
        self._module_dict = module_dict
        self._subset_modules = subset

    def _initialize_modules(
            self,
            module_dict: dict[str, Callable] | None = None,
            subset: Iterable[str] | str | None = None,
    ) -> None:
        module_dict = all_module_dict if module_dict is None else module_dict

        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            _module_dict = {}
            for s in subset:
                if s not in module_dict:
                    raise ValueError(f'Check function {s} is not in default check functions ({module_dict.keys()})')
                else:
                    _module_dict[s] = module_dict[s]
        else:
            _module_dict = module_dict

        self.module_name = []
        self.module_func = []
        self.module_args = []
        self.fname = []
        for module in self.config["modules"]:
            if module["function"] not in _module_dict:
                continue
            function = _module_dict[module["function"]]
            parameters = module.get("parameters", {})
            module_args = set(inspect.signature(function).parameters
                              ).intersection({"mol_pred", "mol_true", "mol_cond"})

            self.module_name.append(module["name"])
            self.fname.append(module["function"])
            self.module_func.append(partial(function, **parameters))
            self.module_args.append(module_args)

    def _run(self) -> Generator[dict, None, None]:
        """Run all tests on molecules provided in file paths.

        Yields:
            Generator of result dictionaries.
        """
        self._initialize_modules(self._module_dict, self._subset_modules)

        for i, paths in self.file_paths.iterrows():
            mol_args = {}
            if "mol_cond" in paths and paths["mol_cond"] is not None:
                mol_cond_load_params = self.config.get("loading", {}).get("mol_cond", {})
                mol_args["mol_cond"] = safe_load_mol(path=paths["mol_cond"], **mol_cond_load_params)
            if "mol_true" in paths and paths["mol_true"] is not None:
                mol_true_load_params = self.config.get("loading", {}).get("mol_true", {})
                mol_args["mol_true"] = safe_load_mol(path=paths["mol_true"], **mol_true_load_params)

            mol_pred_load_params = self.config.get("loading", {}).get("mol_pred", {})
            for i, mol_pred in enumerate(_safe_supply_mols(paths["mol_pred"], **mol_pred_load_params)):
                if self.config["top_n"] is not None and i >= self.config["top_n"]:
                    break

                mol_args["mol_pred"] = mol_pred
                print(str(paths["mol_pred"]))
                results_key = (str(paths["mol_pred"]), self._get_name(mol_pred, i))

                for name, fname, func, args in zip(self.module_name, self.fname, self.module_func, self.module_args):
                    # pick needed arguments for module
                    args = {k: v for k, v in mol_args.items() if k in args}
                    # loading takes all inputs
                    if fname == "loading":
                        args = {k: args.get(k, None) for k in args}
                    # run module when all needed input molecules are valid Mol objects
                    if fname != "loading" and not all(args.get(m, None) for m in args):
                        module_output: dict[str, Any] = {"results": {}}
                    else:
                        try:
                            with time_limit(600):
                                module_output = func(**args)
                        except Exception as e:
                            print(f'mol_pred: ', paths["mol_pred"])
                            print(f'mol_cond: ', paths["mol_cond"])
                            print(f'mol_true: ', paths["mol_true"])
                            # raise e
                            module_output: dict[str, Any] = {"results": {}}
                    # print(name, fname, func)
                    # save to object
                    self.results[results_key].extend([(name, k, v) for k, v in module_output["results"].items()])
                    # self.results[results_key]["details"].append(module_output["details"])

                # return results for this entry
                yield {results_key: self.results[results_key]}

    def clean_up(self):
        self.results: dict[tuple[str, str], list[tuple[str, str, Any]]] = defaultdict(list)

    def bust(
            self,
            mol_pred: Iterable[Mol | Path] | Mol | Path,
            mol_true: Mol | Path | None = None,
            mol_cond: Mol | Path | None = None,
            mol_cond_block: str | None = None,
            full_report: bool = False,
    ) -> pd.DataFrame:
        """Run tests on one or more molecules.

        Args:
            mol_pred: Generated molecule(s), e.g. de-novo generated molecule or docked ligand, with one or more poses.
            mol_true: True molecule, e.g. crystal ligand, with one or more poses.
            mol_cond: Conditioning molecule, e.g. protein.
             mol_cond_block: Conditioning molecule string block, e.g. protein PDB String.
            full_report: Whether to include all columns in the output or only the boolean ones specified in the config.

        Notes:
            - Molecules can be provided as rdkit molecule objects or file paths.

        Returns:
            Pandas dataframe with results.
        """
        if mol_cond_block is not None:
            mol_cond_load_params = self.config.get("loading", {}).get("mol_cond", {})
            mol_cond = MolFromPDBBlock(
                mol_cond_block,
                sanitize=False,
                removeHs=mol_cond_load_params.get("removeHs", False),
                proximityBonding=mol_cond_load_params.get("proximityBonding", False),
            )
            if mol_cond is not None:
                mol_cond.SetProp("_Path", "UNK_PATH")
            mol_cond = _process_mol(
                mol_cond,
                cleanup=mol_cond_load_params.get("cleanup", False),
                sanitize=mol_cond_load_params.get("sanitize", False),
                add_hs=mol_cond_load_params.get("add_hs", False),
                assign_stereo=mol_cond_load_params.get("assign_stereo", False),
            )

        mol_pred = [mol_pred] if isinstance(mol_pred, (Mol, Path, str)) else mol_pred

        columns = ["mol_pred", "mol_true", "mol_cond"]
        mol_table = pd.DataFrame([[mol_pred, mol_true, mol_cond] for mol_pred in mol_pred], columns=columns)

        return self.bust_table(mol_table, full_report)

def bust(
        mol_pred: list[Path | Mol] = [],
        mol_true: Path | Mol | None = None,
        mol_cond: Path | Mol | None = None,
        table: Path | None = None,
        outfmt: str = "short",
        output=sys.stdout,
        full_report_output=sys.stdout,
        config: Path | None = None,
        no_header: bool = False,
        full_report: bool = False,
        top_n: int | None = None,
):
    """PoseBusters: Plausibility checks for generated molecule poses."""
    if table is None and len(mol_pred) == 0:
        raise ValueError("Provide either MOLS_PRED or TABLE.")
    elif table is not None:
        # run on table
        file_paths = pd.read_csv(table, index_col=None)
        file_paths.rename(
            {
                'protein_pdb': 'mol_cond',
                'ligand': 'mol_true',
                'docked_lig': 'mol_pred',
            },
            axis = 1, inplace = True,
        )
        mode = _select_mode(config, file_paths.columns.tolist())
        posebusters = PoseChecker(mode, top_n=top_n)
        posebusters.file_paths = file_paths
        posebusters_results = posebusters._run()
    else:
        # run on single input
        d = {k for k, v in dict(mol_pred=mol_pred, mol_true=mol_true, mol_cond=mol_cond).items() if v}
        mode = _select_mode(config, d)
        posebusters = PoseChecker(mode, top_n=top_n)
        cols = ["mol_pred", "mol_true", "mol_cond"]
        posebusters.file_paths = pd.DataFrame([[mol_pred, mol_true, mol_cond] for mol_pred in mol_pred], columns=cols)
        posebusters_results = posebusters._run()

    for i, results_dict in enumerate(posebusters_results):
        if not full_report:
            full_report_results = _dataframe_from_output(copy.deepcopy(results_dict), posebusters.config, True)
            full_report_output.write(_format_results(full_report_results, outfmt, no_header, i))

        results = _dataframe_from_output(results_dict, posebusters.config, full_report)
        output.write(_format_results(results, outfmt, no_header, i))
    return


if __name__ == '__main__':
    def _path(path_str: str):
        path = Path(path_str)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"File {path} not found!")
        return path

    parser = argparse.ArgumentParser(
        description = "Docked Pose Checker based on PoseBusters test suite.",
        add_help = True,
    )
    parser.add_argument(
        'input_csv',
        type=_path,
        help='The csv file containing ligand, docked_lig and protein_pdb columns. '
             'The ligand should be crystal ligand.',
    )
    parser.add_argument(
        '-n',
        '--lib_expected_number',
        type=int, default=None,
        help='The number of system of benchmark lib.',
    )
    args = parser.parse_args()

    parent = args.input_csv.parent
    stem = args.input_csv.stem
    with open(str(parent / (stem + '_summary.csv')), 'w') as fw, \
            open(str(parent / (stem + '_full_report.csv')), 'w') as full_report_fw:
        bust(
            table=args.input_csv,
            outfmt="csv",
            output=fw,
            full_report_output=full_report_fw,
            full_report=False,
        )

    from DiffBindFR.evaluation import report_pb
    full_report = pd.read_csv(parent / (stem + '_full_report.csv'))
    full_report.rename(
        {'file': 'docked_lig'},
        axis = 1, inplace = True,
    )
    df = pd.read_csv(args.input_csv)
    df = df.merge(full_report, on = ['docked_lig'])
    df.drop('l-rmsd', axis = 1, inplace = True)
    df.rename({'rmsd': 'l-rmsd'}, axis=1, inplace=True)
    pocket_index = 'center' if 'center' in df.columns else 'crystal_ligand'
    if 'smina_score' in df.columns:
        print('Smina Top1 PB validity...')
        top1_df = df.loc[
            df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['smina_score'].agg('idxmin')
        ].reset_index(drop=True)
        report_pb(top1_df, expected_pose_number=args.lib_expected_number)
        print()

    if 'mdn_score' in df.columns:
        print('MDN Top1 PB validity...')
        top1_df = df.loc[
            df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['mdn_score'].agg('idxmax')
        ].reset_index(drop=True)
        report_pb(top1_df, expected_pose_number=args.lib_expected_number)
        print()

    print('PoseBusters test is done!')