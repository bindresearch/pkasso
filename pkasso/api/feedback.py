from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .config import FEEDBACK_DB


MAX_SMILES_LEN = 4000
MAX_COMMENT_LEN = 8000


def clean_feedback(smiles: str, comment: str) -> tuple[str, str]:
    smiles = smiles.strip()
    comment = comment.strip()
    if not smiles:
        raise ValueError("Please enter a SMILES code.")
    if not comment:
        raise ValueError("Please enter a comment.")
    if len(smiles) > MAX_SMILES_LEN:
        raise ValueError("SMILES code is too long.")
    if len(comment) > MAX_COMMENT_LEN:
        raise ValueError("Comment is too long.")
    return smiles, comment


def save_feedback(smiles: str, comment: str) -> None:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with sqlite3.connect(FEEDBACK_DB) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smiles TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            "INSERT INTO feedback (smiles, comment, created_at) VALUES (?, ?, ?)",
            (smiles, comment, created_at),
        )
