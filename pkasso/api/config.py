from __future__ import annotations

import os
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent
STATIC_DIR = API_ROOT / "static"
TEMPLATE_DIR = API_ROOT / "templates"
DATA_DIR = API_ROOT / "data"
FEEDBACK_DIR = Path(os.environ.get("PKASSO_PATH_FEEDBACK", DATA_DIR))
FEEDBACK_DB = FEEDBACK_DIR / "feedback.sqlite3"

DEFAULT_LIGAND = "Example"
DEFAULT_SMILES = "c1cc(C(=O)O)ccc1CNCC(N)C(=O)O"
CUTOFF_STATES = 200
_UNIPKA_MODEL_FOLDER = os.environ.get("PKASSO_UNIPKA_MODEL_FOLDER") or os.environ.get(
    "UNIPKA_MODEL_FOLDER"
)
UNIPKA_MODEL_FOLDER = Path(_UNIPKA_MODEL_FOLDER) if _UNIPKA_MODEL_FOLDER else None
