from __future__ import annotations

import os
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent
STATIC_DIR = API_ROOT / "static"
TEMPLATE_DIR = API_ROOT / "templates"
DATA_DIR = API_ROOT / "data"
FEEDBACK_DIR = Path(os.environ.get("PKASSO_PATH_FEEDBACK", DATA_DIR))
FEEDBACK_DB = FEEDBACK_DIR / "feedback.sqlite3"

DEFAULT_LIGAND = "mymolecule"
DEFAULT_SMILES = "OC(=O)C(c1ccc(O)cc1)CNCCN"
CUTOFF_STATES = 200
