from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_LIGAND, DEFAULT_SMILES


@dataclass
class AppState:
    ligand: str = DEFAULT_LIGAND
    smiles: str = DEFAULT_SMILES
    ph: float = 7.0
    nmols_export: int = 3
    tautomer_search: bool = True
    scan_enabled: bool = False
    smiles_out: list[str] = field(default_factory=list)
    mols_out: list[Any] = field(default_factory=list)
    scan: Any | None = None
    scan_figures: dict[int, str] = field(default_factory=dict)
    error: str | None = None


SESSIONS: dict[str, AppState] = {}


def update_state_from_form(state: AppState, form: dict[str, str]) -> None:
    state.ligand = form.get("ligand", DEFAULT_LIGAND).strip() or DEFAULT_LIGAND
    state.smiles = form.get("smiles", DEFAULT_SMILES).strip()
    state.tautomer_search = form.get("tautomer_search") == "on"
    state.scan_enabled = form.get("scan_enabled") == "on"

    try:
        state.ph = max(0.0, min(14.0, float(form.get("ph", "7.0"))))
    except ValueError:
        state.ph = 7.0

    try:
        state.nmols_export = max(1, min(20, int(form.get("nmols_export", "3"))))
    except ValueError:
        state.nmols_export = 3
