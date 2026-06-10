from __future__ import annotations

from typing import TypedDict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


IMIDIC_ACID_PATTERN = Chem.MolFromSmarts(
    "[NX2;!$([N+])]=[CX3]([OX2H1])"
)
THIOIMIDIC_ACID_PATTERN = Chem.MolFromSmarts(
    "[NX2;!$([N+])]=[CX3]([SX2H1])"
)
HYDROXAMATE_PATTERN = Chem.MolFromSmarts(
    "[CX3](=[OX1])-[NX3]-[OX2H1]"
)
HYDROXIMIC_ACID_PATTERN = Chem.MolFromSmarts(
    "[CX3]([OX2H1])=[NX2]-[OX2H1]"
)


ConformerEnergy = tuple[int, float]


class TautomerEntry(TypedDict):
    """Store a prepared tautomer and its ranking data."""

    idx: int
    taut: Chem.Mol
    mol3d: Chem.Mol
    conf_energies: list[ConformerEnergy]
    mmff_energy: float
    rdkit_score: int


def has_imidic_acid_amide_tautomer(mol: Chem.Mol) -> bool:
    """Return whether a molecule matches an imidic or thioimidic acid pattern."""

    return bool(
        mol.HasSubstructMatch(IMIDIC_ACID_PATTERN)
        or mol.HasSubstructMatch(THIOIMIDIC_ACID_PATTERN)
    )


def has_hydroxamate_tautomer(mol: Chem.Mol) -> bool:
    """Return whether a molecule matches the hydroxamate tautomer pattern."""

    return bool(mol.HasSubstructMatch(HYDROXAMATE_PATTERN))


def has_hydroximic_acid_tautomer(mol: Chem.Mol) -> bool:
    """Return whether a molecule matches the hydroximic acid tautomer pattern."""

    return bool(mol.HasSubstructMatch(HYDROXIMIC_ACID_PATTERN))


def rdkit_tautomer_conformers(
    mol: Chem.Mol,
    num_confs: int = 10,
) -> tuple[Chem.Mol, list[ConformerEnergy]] | None:
    """Generate and MMFF-rank conformers for a tautomer."""

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()

    params.useRandomCoords = False

    conf_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        params=params,
    )

    if len(conf_ids) == 0:
        return None

    # MMFF optimization
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)

    if mmff_props is None:
        return None

    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol,
        mmffVariant="MMFF94s",
    )

    conf_energies: list[ConformerEnergy] = []

    for conf_id, (_, energy) in zip(conf_ids, results):
        conf_energies.append((conf_id, energy))

    conf_energies.sort(key=lambda x: x[1])

    return mol, conf_energies


def best_tautomer_smiles(
    smiles: str,
    max_tautomers: int = 20,  # max number of tautomers for rdkit
    num_confs: int = 10,  # conformations per tautomer (rdkit)
) -> str:
    """Return a chemically plausible tautomer SMILES using RDKit and MMFF ranking."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return smiles

    enumerator = rdMolStandardize.TautomerEnumerator()

    tautomers = list(enumerator.Enumerate(mol))

    if len(tautomers) == 1:
        return str(Chem.MolToSmiles(tautomers[0]))

    if len(tautomers) > max_tautomers:
        print("Exceeding max tautomers, using input smiles.")
        return smiles

    # ---------------------------------------------------------
    # Stage 1:
    # Generate conformers + MMFF pre-ranking
    # ---------------------------------------------------------

    ranked: list[TautomerEntry] = []

    for i, taut in enumerate(tautomers):

        try:

            Chem.AssignStereochemistry(
                taut,
                force=True,
                cleanIt=True,
            )

            prep = rdkit_tautomer_conformers(
                taut,
                num_confs=num_confs,
            )

            if prep is None:
                continue

            mol3d, conf_energies = prep

            ranked.append(
                {
                    "idx": i,
                    "taut": taut,
                    "mol3d": mol3d,
                    "conf_energies": conf_energies,
                    "mmff_energy": conf_energies[0][1],
                    "rdkit_score": enumerator.ScoreTautomer(taut),
                }
            )

        except Exception as e:
            print(f"Skipping tautomer {i}: {e}")

    if len(ranked) == 0:
        return smiles

    # ---------------------------------------------------------
    # Stage 2:
    # RDKit tautomer-score + MMFF filter
    # ---------------------------------------------------------

    if any(has_hydroxamate_tautomer(entry["taut"]) for entry in ranked):
        ranked = [
            entry
            for entry in ranked
            if not has_hydroximic_acid_tautomer(entry["taut"])
        ]

    if len(ranked) == 0:
        return smiles

    # Filter against imidic acid
    best_rdkit_score = max(entry["rdkit_score"] for entry in ranked)
    ranked = [
        entry
        for entry in ranked
        if (
            entry["rdkit_score"] == best_rdkit_score
            or not has_imidic_acid_amide_tautomer(entry["taut"])
        )
    ]

    ranked.sort(key=lambda x: (-x["rdkit_score"], x["mmff_energy"]))

    if len(ranked) == 0:
        return smiles
    else:
        return str(Chem.MolToSmiles(ranked[0]["taut"]))
