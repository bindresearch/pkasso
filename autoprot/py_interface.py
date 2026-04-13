"""High-level Python interface for running AutoProt predictions."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore

from .main import Autoprot
from .postprocess import Batch, Molecule, Scan

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
        smiles_dict: dict[str, str],
        pH: float = 7.0,
        **kwargs: Any) -> Batch:
    """
    Batch process a dict of names and smiles input files.

    Use:
    ```
    from autoprot import batch_protonate

    batch_input = {
        'fasudil' : 'C1CNCCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3',
        'mymolecule' : 'OC(=O)C(c1ccc(O)cc1)CNCCN',
        'histamine' : 'C1=C(NC=N1)CCN',
    }

    batch = batch_protonate(batch_input, pH=7., cutoff_export=0.2)

    for name, molecule in batch.molecules.items():
        print(name, molecule.smiles)
    ```
    """

    batch_dict: dict[str, Molecule] = {}

    for name, smiles in tqdm(smiles_dict.items(),total=len(smiles_dict)):
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
