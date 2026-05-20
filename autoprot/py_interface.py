"""High-level Python interface for running AutoProt predictions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore
from rdkit.Chem import MolToSmiles
from rdkit.Chem.rdchem import Mol

from .main import Autoprot
from .postprocess import Molecule, Scan

def protonate(inp: str | Mol, pH: float = 7.0, **kwargs: Any) -> tuple[list[str], list[Mol]]:
    """
    Helper function to run autoprot via:

    ```
    from autoprot import protonate

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

    ap = Autoprot(
        smiles, **kwargs)
    molecule = ap.run_single(pH=pH)

    return molecule.smiles, molecule.mols

def batch_protonate(
        input_list: list[str | Mol], # dict[str, str],
        pH: float = 7.0,
        **kwargs: Any) -> tuple[list[list[str]], list[list[Mol]]]:
    """
    Batch process a list of smiles or a list of rdkit Mol objects.

    Use:
    ```
    from autoprot import batch_protonate

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
        ap = Autoprot(
            inp, **kwargs) # name=name,
        ap.run_single(pH=pH)

        batch_smiles.append(ap.molecule.smiles)
        batch_mols.append(ap.molecule.mols)

    return batch_smiles, batch_mols

def scan_pH(
        inp: str | Mol,
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
        inp, **kwargs)
    return ap.run_scan(pHs=pHs_arr)
