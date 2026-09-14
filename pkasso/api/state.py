from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_LIGAND, DEFAULT_SMILES


@dataclass
class AppState:
    ligand: str = DEFAULT_LIGAND
    smiles: str = DEFAULT_SMILES
    model: str = "molgpka"
    ph: float = 7.0
    nmols_export: int = 3
    tautomer_search: bool = True
    smiles_out: list[str] = field(default_factory=list)
    mols_out: list[Any] = field(default_factory=list)
    scan: Any | None = None
    scan_figures: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


SESSIONS: dict[str, AppState] = {}


def update_state_from_form(state: AppState, form: dict[str, str]) -> None:
    state.ligand = form.get("ligand", DEFAULT_LIGAND).strip() or DEFAULT_LIGAND
    state.smiles = form.get("smiles", DEFAULT_SMILES).strip()
    state.tautomer_search = form.get("tautomer_search") == "on"
    model = form.get("model", "molgpka")
    state.model = model if model in {"molgpka", "unipka", "mixed"} else "molgpka"


def update_microstate_selection(state: AppState, ph: str, nmols_export: str) -> None:
    """Validate controls used to select lazy single-pH output."""

    try:
        state.ph = max(0.0, min(14.0, float(ph)))
    except ValueError:
        state.ph = 7.0

    try:
        state.nmols_export = max(1, min(20, int(nmols_export)))
    except ValueError:
        state.nmols_export = 3
