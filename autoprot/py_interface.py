from .main import Autoprot
from .postprocess import Molecule, Microstate, Batch, Scan
from .utils import *

from dataclasses import dataclass
from tqdm import tqdm # type: ignore

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from typing import Any

def protonate(smiles: str, pH: float = 7.0, **kwargs: Any) -> Molecule:
    """
    Helper function to run autoprot via:

    ```
    from autoprot import protonate

    name = 'mymolecule'
    smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
    pH = 7.0
    cutoff_export = 0.2 

    molecule = protonate(smiles, name=name, pH=pH, cutoff_export=cutoff_export)
    ```
    """

    ap = Autoprot(
        smiles, **kwargs)
    ap.run_single(pH=pH)

    return ap.molecule

def batch_protonate(
        batch_file: str,
        pH: float = 7.0,
        **kwargs: Any) -> Batch:
    """
    Batch process a .smi input file.

    Use:
    ```
    from autoprot import batch_protonate

    batch_file = 'example_molecules.smi'
    batch = batch_protonate(batch_file, pH=7., cutoff_export=0.2)

    for name, molecule in batch.molecules.items():
        print(name, molecule.smiles)
    ```
    """

    print(batch_file)
    names_batch, smiles_batch = read_smi(batch_file)

    # mols: dict[str, Mol] = {}
    batch_dict: dict[str, Molecule] = {}

    for name, smiles in tqdm(zip(names_batch, smiles_batch),total=len(names_batch)):
        ap = Autoprot(
            smiles, name=name, **kwargs)
        ap.run_single(pH=pH)

        batch_dict[name] = ap.molecule
    batch = Batch(batch_dict)
    return batch

def scan_pH(
        smiles: str,
        pHs: NDArray[np.float64] | list[float] = np.arange(0, 14.1, 0.25, dtype=np.float64), 
        **kwargs: Any
) -> Scan:
    """
    Run autoprot pH scan

    ```
    from autoprot import scan_pH

    smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
    name = 'mymolecule'

    scan = scan_pH(
        smiles,
        name = name,
    )

    scan.print_macro_pkas()
    scan.plot_scan()
    scan.plot_mols()
    ```
    """

    pHs_arr: NDArray[np.float64] = np.array(pHs)

    ap = Autoprot(
        smiles, **kwargs)
    ap.run_scan(pHs=pHs_arr)

    scan = Scan(
        ap.name,
        ap.indices0,
        ap.state_strs_relevant,
        ap.mols_relevant,
        ap.sfreqs_relevant,
        ap.pHs,
        ap.net_charges_arr,
        ap.sfreqs_not_relevant,
        ap.pkas_macro,
    )
    return scan