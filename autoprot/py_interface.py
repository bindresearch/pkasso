from .main import Autoprot
from .postprocess import Molecule, Microstate, Batch, Scan
from .utils import *

from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from typing import Any

def protonate(smiles: str, pH: float = 7.0, **kwargs) -> Microstate:
    """
    Helper function to run autoprot via
    from autoprot import protonate
    
    smiles_input = 'OC(=O)C(c1ccc(O)cc1)CNCCN'
    mols, smiles, freqs = protonate(smiles_input)
    """
    # kwargs['write_output'] = kwargs.get('write_output',False)

    ap = Autoprot(
        smiles, **kwargs)
    ap.run_single(pH=pH)

    return ap.molecule

def batch_protonate(
        batch_file: str,
        pH: float = 7.0,
        **kwargs) -> dict[str, Microstate]:
    """
    Batch process a .smi input file.
    """

    print(batch_file)
    names_batch, smiles_batch = read_smi(batch_file)

    # mols: dict[str, Mol] = {}
    batch_dict: dict[str, dict[str, Any]] = {}

    for name, smiles in tqdm(zip(names_batch, smiles_batch),total=len(names_batch)):
        ap = Autoprot(
            smiles, name=name, **kwargs)
        ap.run_single(pH=pH)

        batch_dict[name] = ap.molecule
    batch = Batch(batch_dict)
    return batch

def scan_pH(
        smiles: str, 
        pHs: NDArray[np.float64] | list[float] = np.arange(0, 14.1, 0.25), 
        **kwargs
) -> Microstate: # FIX OUTPUT TYPING
    """
    Run autoprot pH scan

    smiles_input = 'OC(=O)C(c1ccc(O)cc1)CNCCN'
    mols, smiles, freqs = protonate(smiles_input)
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