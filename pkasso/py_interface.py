"""High-level Python interface for running pKasso predictions."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from rdkit.Chem import MolToSmiles
from rdkit.Chem.rdchem import Mol

from .main import pKasso
from .predict_pka import PredictorKey, resolve_predictor_cls
from .postprocess import Scan

ModelInput = PredictorKey | str | Sequence[PredictorKey | str]


def _resolve_model_kwargs(
    kwargs: dict[str, Any],
    model: ModelInput,
) -> dict[str, Any]:
    """Resolve public model keys before constructing pKasso."""

    pkasso_kwargs = dict(kwargs)
    if "pka_predictor_cls" in pkasso_kwargs or "pka_predictor_classes" in pkasso_kwargs:
        raise ValueError("Python entry points accept model keys. Pass predictor classes directly to pKasso.")

    if isinstance(model, str):
        pkasso_kwargs["pka_predictor_cls"] = resolve_predictor_cls(model)
        return pkasso_kwargs

    model_classes = tuple(resolve_predictor_cls(model_key) for model_key in model)
    if not model_classes:
        raise ValueError("At least one model key is required.")
    if len(model_classes) == 1:
        pkasso_kwargs["pka_predictor_cls"] = model_classes[0]
    else:
        pkasso_kwargs["pka_predictor_classes"] = model_classes
    return pkasso_kwargs


def protonate(
    inp: str | Mol,
    pH: float = 7.0,
    model: ModelInput = "molgpka",
    **kwargs: Any,
) -> tuple[tuple[str, ...], tuple[Mol, ...]]:
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

    ap = pKasso(smiles, **_resolve_model_kwargs(kwargs, model))
    molecule = ap.run_single(pH=pH)

    return molecule.smiles, molecule.mols


def batch_protonate(
        input_list: list[str | Mol],
        pH: float = 7.0,
        model: ModelInput = "molgpka",
        **kwargs: Any
) -> tuple[list[tuple[str, ...]], list[tuple[Mol, ...]]]:
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

    batch_smiles: list[tuple[str, ...]] = []
    batch_mols: list[tuple[Mol, ...]] = []

    for inp in tqdm(input_list):
        ap = pKasso(inp, **_resolve_model_kwargs(kwargs, model))
        molecule = ap.run_single(pH=pH)

        batch_smiles.append(molecule.smiles)
        batch_mols.append(molecule.mols)

    return batch_smiles, batch_mols


def scan_pH(
    inp: str | Mol,
    pHs: NDArray[np.float64] | list[float] = np.arange(0, 14.1, 0.25, dtype=np.float64),
    model: ModelInput = "molgpka",
    **kwargs: Any,
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

    ap = pKasso(inp, **_resolve_model_kwargs(kwargs, model))
    return ap.run_scan(pHs=pHs_arr)
