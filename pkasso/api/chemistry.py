from __future__ import annotations

import copy
import io
import logging
import numpy as np
import os
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any
from rdkit.Chem import AllChem
from rdkit import Chem

from .config import CUTOFF_STATES, UNIPKA_MODEL_FOLDER
from .state import AppState

from ..py_interface import scan_pH, protonate
from ..postprocess import draw_mols

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pkasso")


class _WarningCollector(logging.Handler):
    def __init__(self, messages: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_pkasso_warnings() -> Iterator[list[str]]:
    """Collect warnings emitted by the core workflow while preserving normal logging."""

    messages: list[str] = []
    handler = _WarningCollector(messages)
    core_logger = logging.getLogger("pkasso.main")
    core_logger.addHandler(handler)
    try:
        yield messages
    finally:
        core_logger.removeHandler(handler)


def _unique(messages: list[str]) -> list[str]:
    return list(dict.fromkeys(messages))


def prediction_kwargs(state: AppState) -> dict[str, Any]:
    if not state.precision_mode:
        return {}

    unipka_options: dict[str, object] = {
        "folds": (2,),
    }
    if UNIPKA_MODEL_FOLDER is not None:
        unipka_options["model_dir"] = UNIPKA_MODEL_FOLDER

    return {
        "total_max_sites": 8,
        "nthreads": 0,
        "model": {
            "molgpka": {},
            "unipka": unipka_options,
        },
    }


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_svg_image(svg: str | bytes) -> str:
    if isinstance(svg, bytes):
        return svg.decode("utf-8")
    return svg


def figure_to_svg(fig: Any) -> str:
    plt = _pyplot()
    if fig is None:
        fig = plt.gcf()
    elif hasattr(fig, "figure") and not hasattr(fig, "savefig"):
        fig = fig.figure

    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def compute_prediction(state: AppState) -> None:

    state.error = None
    state.warnings.clear()
    captured_warnings: list[str] = []
    try:
        with capture_pkasso_warnings() as captured_warnings:
            smiles_out, mols_out = protonate(
                state.smiles,
                name=state.ligand,
                pH=state.ph,
                cutoff_export=0.0,
                cutoff_states=CUTOFF_STATES,
                tautomer_search=state.tautomer_search,
                free_energy_cutoff_individual = 10,
                free_energy_cutoff_combined = 10,
                **prediction_kwargs(state),
            )
    finally:
        state.warnings = _unique(captured_warnings)
    state.smiles_out = list(smiles_out[: state.nmols_export])
    state.mols_out = list(mols_out[: state.nmols_export])
    state.scan = None
    state.scan_figures.clear()


def compute_scan(state: AppState) -> None:

    state.error = None
    state.warnings.clear()
    captured_warnings: list[str] = []
    try:
        with capture_pkasso_warnings() as captured_warnings:
            state.scan = scan_pH(
                state.smiles,
                name=state.ligand,
                cutoff_states=CUTOFF_STATES,
                tautomer_search=state.tautomer_search,
                free_energy_cutoff_individual = 100,
                free_energy_cutoff_combined = 100,
                pHs=np.arange(0, 14.1, 0.25, dtype=np.float64),
                **prediction_kwargs(state),
            )
    finally:
        state.warnings = _unique(captured_warnings)
    state.scan_figures.clear()


def draw_molecule_grid(mols: list[Any], show_probability: bool = True) -> str:

    if not mols:
        return ""

    svg = draw_mols(
        mols,
        subImgSize=(400, 350),
        max_cols=3,
        show_probability=show_probability,
    )
    return render_svg_image(svg)


def draw_single_molecule(mol: Any) -> str:

    svg = draw_mols([mol], subImgSize=(520, 430), max_cols=1, show_probability=False)
    return render_svg_image(svg)


def scan_figure_svg(state: AppState, highlight_idx: int) -> str | None:

    if state.scan is None:
        return None

    max_idx = len(getattr(state.scan, "mols_relevant", []))
    highlight_idx = max(0, min(max_idx, highlight_idx))
    if highlight_idx not in state.scan_figures:
        fig = state.scan.plot_scan(highlight_idx=highlight_idx)
        state.scan_figures[highlight_idx] = figure_to_svg(fig)
    return state.scan_figures[highlight_idx]


def sdf_for_state(state: AppState) -> bytes:

    sdf = ""
    for mol in state.mols_out:
        mol2 = copy.deepcopy(mol)
        mol2 = Chem.AddHs(mol2, addCoords=True)
        AllChem.EmbedMolecule(mol2, randomSeed=1, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol2)
        sdf += Chem.MolToMolBlock(mol2) + "\n$$$$\n"
    return sdf.encode("utf-8")
