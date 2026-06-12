"""High-level Python interface for running pKasso predictions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from rdkit.Chem import MolToSmiles
from rdkit.Chem.rdchem import Mol

from .main import pKasso
from .postprocess import Scan


def protonate(inp: str | Mol, pH: float = 7.0, **kwargs: Any) -> tuple[list[str], list[Mol]]:
    """
    Helper function to run pkasso via:

    ```
    from pkasso import protonate

    name = 'mymolecule'
    smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
    pH = 7.0
    cutoff_export = 0.2

    smiles, mols = protonate(smiles, name=name, pH=pH, cutoff_export=cutoff_export)
    ```
    """

    if isinstance(inp, Mol):
        smiles = MolToSmiles(inp)
    else:
        smiles = inp

    ap = pKasso(smiles, **kwargs)
    molecule = ap.run_single(pH=pH)

    return molecule.smiles, molecule.mols


def batch_protonate(input_list: list[str | Mol], pH: float = 7.0, **kwargs: Any) -> tuple[list[list[str]], list[list[Mol]]]:
    """
    Batch process a list of smiles or a list of rdkit Mol objects.

    Use:
    ```
    from pkasso import batch_protonate

    batch_input = [
        'C1CNCCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3',
        'OC(=O)C(c1ccc(O)cc1)CNCCN',
        'C1=C(NC=N1)CCN',
    ]

    smiles_out, mols_out = batch_protonate(batch_input, pH=7., cutoff_export=0.2)
    ```
    """

    batch_smiles: list[list[str]] = []
    batch_mols: list[list[Mol]] = []

    for inp in tqdm(input_list):
        ap = pKasso(inp, **kwargs)
        molecule = ap.run_single(pH=pH)

        batch_smiles.append(molecule.smiles)
        batch_mols.append(molecule.mols)

    return batch_smiles, batch_mols


def scan_pH(
    inp: str | Mol, pHs: NDArray[np.float64] | list[float] = np.arange(0, 14.1, 0.25, dtype=np.float64), **kwargs: Any
) -> Scan:
    """
    Run pkasso pH scan

    ```
    from pkasso import scan_pH

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

    ap = pKasso(inp, **kwargs)
    return ap.run_scan(pHs=pHs_arr)
