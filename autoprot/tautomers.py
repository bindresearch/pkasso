import os
import re
import math
import tempfile
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


HARTREE_PER_KCAL_MOL = 1.0 / 627.5094740631
GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
DEFAULT_TEMPERATURE_K = 298.15
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

def has_imidic_acid_amide_tautomer(mol):
    return (
        mol.HasSubstructMatch(IMIDIC_ACID_PATTERN)
        or mol.HasSubstructMatch(THIOIMIDIC_ACID_PATTERN)
    )

def has_hydroxamate_tautomer(mol):
    return mol.HasSubstructMatch(HYDROXAMATE_PATTERN)

def has_hydroximic_acid_tautomer(mol):
    return mol.HasSubstructMatch(HYDROXIMIC_ACID_PATTERN)

def rdkit_tautomer_conformers(
    mol,
    num_confs=10,
    # random_seed=0xF00D,
):

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()

    # Deterministic embeddings
    # params.randomSeed = random_seed
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

    conf_energies = []

    for conf_id, (_, energy) in zip(conf_ids, results):
        conf_energies.append((conf_id, energy))

    conf_energies.sort(key=lambda x: x[1])

    return mol, conf_energies

def best_tautomer_smiles(
    smiles,
    max_tautomers: int = 20, # max number of tautomers for rdkit
    num_confs: int = 10, # conformations per tautomer (rdkit)
):
    """Return a chemically plausible low-energy tautomer.

    RDKit's tautomer score is used to rank tautomers followed by filtering.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return smiles

    enumerator = rdMolStandardize.TautomerEnumerator()

    tautomers = list(enumerator.Enumerate(mol))

    if len(tautomers) == 1:
        return Chem.MolToSmiles(tautomers[0])

    if len(tautomers) > max_tautomers:
        print("Exceeding max tautomers, using input smiles.")
        return smiles

    # ---------------------------------------------------------
    # Stage 1:
    # Generate conformers + MMFF pre-ranking
    # ---------------------------------------------------------

    ranked = []

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
        return Chem.MolToSmiles(ranked[0]["taut"])