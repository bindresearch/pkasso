"""High-level Python interface for running pKasso predictions."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from rdkit.Chem import MolToSmiles
from rdkit.Chem.rdchem import Mol

from .main import pKasso
from .predict_pka import ModelInput
from .postprocess import Scan


def _resolve_model_kwargs(
    kwargs: dict[str, Any],
    model: ModelInput | None,
) -> dict[str, Any]:
    """Pass the public model mapping to pKasso."""

    pkasso_kwargs = dict(kwargs)
    if "pka_predictor_cls" in pkasso_kwargs or "pka_predictor_classes" in pkasso_kwargs:
        raise ValueError("Python entry points accept a model mapping, not predictor classes.")
    if model is not None:
        pkasso_kwargs["model"] = model
    return pkasso_kwargs


def _validate_prediction_options(pH: float | None, kwargs: dict[str, Any]) -> None:
    """Validate numeric options exposed by the Python entry points."""

    if pH is not None and not math.isfinite(pH):
        raise ValueError("pH must be finite.")

    cutoff_export = kwargs.get("cutoff_export")
    if cutoff_export is not None and (
        not math.isfinite(cutoff_export)
        or cutoff_export < 0
        or cutoff_export > 1
    ):
        raise ValueError("cutoff_export must be between 0 and 1.")


def protonate(
    inp: str | Mol,
    pH: float = 7.0,
    model: ModelInput | None = None,
    nthreads: int = 0,
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

    Select and configure predictors with an ordered mapping, for example
    ``model={"molgpka": {}, "unipka": {"gpu": True}}``.
    ``nthreads`` controls RDKit, MolGpKa, and Uni-pKa CPU thread counts.
    """

    _validate_prediction_options(pH, kwargs)

    if isinstance(inp, Mol):
        smiles = MolToSmiles(inp)
    else:
        smiles = inp

    ap = pKasso(smiles, nthreads=nthreads, **_resolve_model_kwargs(kwargs, model))
    molecule = ap.run_single(pH=pH)

    return molecule.smiles, molecule.mols


def batch_protonate(
        input_list: list[str | Mol],
        pH: float = 7.0,
        model: ModelInput | None = None,
        nthreads: int = 0,
        progress: bool = True,
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

    Set ``progress=False`` to disable the progress bar.
    """

    _validate_prediction_options(pH, kwargs)

    batch_smiles: list[tuple[str, ...]] = []
    batch_mols: list[tuple[Mol, ...]] = []

    for inp in tqdm(input_list, disable=not progress):

        if isinstance(inp, Mol):
            smiles = MolToSmiles(inp)
        else:
            smiles = inp

        ap = pKasso(smiles, nthreads=nthreads, **_resolve_model_kwargs(kwargs, model))
        molecule = ap.run_single(pH=pH)

        batch_smiles.append(molecule.smiles)
        batch_mols.append(molecule.mols)

    return batch_smiles, batch_mols


def scan_pH(
    inp: str | Mol,
    pHs: NDArray[np.float64] | list[float] = np.arange(0, 14.1, 0.25, dtype=np.float64),
    model: ModelInput | None = None,
    nthreads: int = 0,
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

    _validate_prediction_options(None, kwargs)

    if isinstance(inp, Mol):
        smiles = MolToSmiles(inp)
    else:
        smiles = inp

    pHs_arr: NDArray[np.float64] = np.asarray(pHs, dtype=np.float64)
    if pHs_arr.ndim != 1:
        raise ValueError("pHs must be one-dimensional.")
    if pHs_arr.size == 0:
        raise ValueError("pHs must not be empty.")
    if not np.all(np.isfinite(pHs_arr)):
        raise ValueError("pHs must contain only finite values.")

    ap = pKasso(smiles, nthreads=nthreads, **_resolve_model_kwargs(kwargs, model))
    return ap.run_scan(pHs=pHs_arr)
