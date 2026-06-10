from __future__ import annotations

from pathlib import Path


API_ROOT = Path(__file__).resolve().parent
STATIC_DIR = API_ROOT / "static"
TEMPLATE_DIR = API_ROOT / "templates"
DATA_DIR = API_ROOT / "data"
FEEDBACK_DB = DATA_DIR / "feedback.sqlite3"

DEFAULT_LIGAND = "mymolecule"
DEFAULT_SMILES = "OC(=O)C(c1ccc(O)cc1)CNCCN"
CUTOFF_STATES = 200
