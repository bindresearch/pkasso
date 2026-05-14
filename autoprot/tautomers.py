# import os
# import logging
# import subprocess
# import tempfile
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem.MolStandardize import rdMolStandardize
# import re

# logger = logging.getLogger(__name__)

# def mol_to_xyz(mol, filename):
#     conf = mol.GetConformer()
#     with open(filename, "w") as f:
#         f.write(f"{mol.GetNumAtoms()}\n\n")
#         for atom in mol.GetAtoms():
#             pos = conf.GetAtomPosition(atom.GetIdx())
#             f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

# def run_xtb(xyz_file, workdir, optimize=False):

#     env = os.environ.copy()
#     env["OMP_NUM_THREADS"] = "1"

#     if optimize:
#         cmd = ["xtb", xyz_file, "--opt", "--gfn2", "--alpb", "water"] # "--opt",
#     else:
#         cmd = ["xtb", xyz_file, "--gfn2", "--alpb", "water"] # "--opt",

#     result = subprocess.run(cmd, env=env, cwd=workdir, capture_output=True, text=True)

#     energy = None
#     for line in result.stdout.splitlines():
#         if "TOTAL ENERGY" in line:
#             # Extract float using regex (robust to formatting)
#             match = re.search(r"[-+]?\d+\.\d+", line)
#             if match:
#                 energy = float(match.group(0))
#                 break

#     return energy

# def prepare_3d(mol):
#     mol = Chem.AddHs(mol)
#     AllChem.EmbedMolecule(mol, AllChem.ETKDG())
#     AllChem.UFFOptimizeMolecule(mol)
#     return mol

# def best_tautomer_smiles(smiles, max_tautomers: int = 100, xtb_optimize=False):

#     mol = Chem.MolFromSmiles(smiles)

#     enumerator = rdMolStandardize.TautomerEnumerator()
#     tautomers = enumerator.Enumerate(mol)

#     if len(tautomers) == 1:
#         return Chem.MolToSmiles(tautomers[0])
#     if len(tautomers) > max_tautomers:
#         # logger.info('Exceeding max tautomers, using input smiles.')
#         print('Exceeding max tautomers, using input smiles.')
#         return smiles
    
#     best_energy = None
#     best_mol = None

#     for i, taut in enumerate(tautomers):
#         try:
#             taut3d = prepare_3d(Chem.Mol(taut))

#             with tempfile.TemporaryDirectory() as tmpdir:
#                 xyz_path = os.path.join(tmpdir, f"mol_{i}.xyz")
#                 mol_to_xyz(taut3d, xyz_path)

#                 energy = run_xtb(xyz_path, tmpdir, optimize=xtb_optimize)

#             if energy is None:
#                 continue

#             if best_energy is None or energy < best_energy:
#                 best_energy = energy
#                 best_mol = taut

#         except Exception as e:
#             logger.info(f"Skipping tautomer {i}: {e}")

#     if best_mol is None:
#         logger.info('Did not find good tautomer, using input smiles.')
#         return smiles

#     return Chem.MolToSmiles(best_mol)


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


def describe_tautomer_entry(entry):
    flags = []

    if has_imidic_acid_amide_tautomer(entry["taut"]):
        flags.append("imidic")
    if has_hydroxamate_tautomer(entry["taut"]):
        flags.append("hydroxamate")
    if has_hydroximic_acid_tautomer(entry["taut"]):
        flags.append("hydroximic")

    label = ",".join(flags) if flags else "-"
    smiles = Chem.MolToSmiles(entry["taut"])

    return (
        f"idx={entry['idx']} score={entry['rdkit_score']} "
        f"mmff={entry['mmff_energy']:.6f} flags={label} smiles={smiles}"
    )


def print_tautomer_debug(stage, ranked):
    print(f"Tautomer candidates after {stage}:")

    for entry in ranked:
        print(f"  {describe_tautomer_entry(entry)}")


def conformer_ensemble_free_energy(
    energies,
    temperature=DEFAULT_TEMPERATURE_K,
):
    """Boltzmann conformer free energy in the same units as ``energies``."""

    if len(energies) == 0:
        return None

    rt = GAS_CONSTANT_KCAL_MOL_K * temperature * HARTREE_PER_KCAL_MOL
    min_energy = min(energies)
    partition_sum = sum(
        math.exp(-(energy - min_energy) / rt)
        for energy in energies
    )

    return min_energy - rt * math.log(partition_sum)


def run_xtb(xyz_file, workdir, optimize=False):

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    cmd = [
        "xtb",
        xyz_file,
        "--gfn2",
        "--alpb",
        "water",
    ]

    if optimize:
        cmd.insert(2, "--opt")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    energy = None

    for line in result.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            match = re.search(r"[-+]?\d+\.\d+", line)
            if match:
                energy = float(match.group(0))
                break

    return energy


def mol_to_xyz(mol, xyz_path, conf_id=0):

    conf = mol.GetConformer(conf_id)

    with open(xyz_path, "w") as f:

        f.write(f"{mol.GetNumAtoms()}\n")
        f.write("generated by rdkit\n")

        for atom in mol.GetAtoms():

            pos = conf.GetAtomPosition(atom.GetIdx())

            f.write(
                f"{atom.GetSymbol()} "
                f"{pos.x:.6f} "
                f"{pos.y:.6f} "
                f"{pos.z:.6f}\n"
            )


def prepare_tautomer_conformers(
    mol,
    num_confs=4,
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
    max_tautomers_xtb: int = 10, # max number of tautomers for xtb
    num_confs: int = 10, # conformations per tautomer (rdkit)
    use_xtb: bool = False,
    num_xtb_confs: int = 3, # top conformations per tautomer (xtb)
    xtb_optimize: bool = True,
    rdkit_score_window: int = 2,
    temperature: float = DEFAULT_TEMPERATURE_K,
    debug: bool = False,
):
    """Return a chemically plausible low-energy tautomer.

    RDKit's tautomer score is used as a chemical prior before xTB ranking.
    xTB evaluates the lowest ``num_xtb_confs`` MMFF conformers for each
    tautomer surviving the pre-filter, then ranks tautomers by the Boltzmann
    conformer ensemble free energy. Set ``rdkit_score_window=None`` to recover
    pure MMFF/xTB tautomer filtering.
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

            prep = prepare_tautomer_conformers(
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

    if debug:
        print_tautomer_debug("MMFF preparation", ranked)

    # ---------------------------------------------------------
    # Stage 2:
    # RDKit tautomer-score prior + MMFF pre-filter
    # ---------------------------------------------------------

    if any(has_hydroxamate_tautomer(entry["taut"]) for entry in ranked):
        ranked = [
            entry
            for entry in ranked
            if not has_hydroximic_acid_tautomer(entry["taut"])
        ]

    if len(ranked) == 0:
        return smiles

    if rdkit_score_window is not None:
        best_rdkit_score = max(entry["rdkit_score"] for entry in ranked)
        ranked = [
            entry
            for entry in ranked
            if best_rdkit_score - entry["rdkit_score"] <= rdkit_score_window
        ]
        ranked = [
            entry
            for entry in ranked
            if (
                entry["rdkit_score"] == best_rdkit_score
                or not has_imidic_acid_amide_tautomer(entry["taut"])
            )
        ]

    if debug:
        print_tautomer_debug("tautomer filters", ranked)

    ranked.sort(key=lambda x: (-x["rdkit_score"], x["mmff_energy"]))

    ranked = ranked[:max_tautomers_xtb]

    if len(ranked) == 0:
        return smiles

    if len(ranked) == 1:
        return Chem.MolToSmiles(ranked[0]["taut"])

    # ---------------------------------------------------------
    # Stage 3:
    # xTB ranking
    # ---------------------------------------------------------

    if use_xtb:

        best_free_energy = None
        best_taut = None

        num_xtb_confs = max(1, num_xtb_confs)

        for entry in ranked:

            try:

                xtb_energies = []

                with tempfile.TemporaryDirectory() as tmpdir:

                    for conf_id, _ in entry["conf_energies"][:num_xtb_confs]:

                        xyz_path = os.path.join(
                            tmpdir,
                            f"taut_{entry['idx']}_conf_{conf_id}.xyz"
                        )

                        mol_to_xyz(
                            entry["mol3d"],
                            xyz_path,
                            conf_id=conf_id,
                        )
                        # print(f'running xtb on entry {entry["idx"]}')
                        xtb_energy = run_xtb(
                            xyz_path,
                            tmpdir,
                            optimize=xtb_optimize,
                        )

                        if xtb_energy is None:
                            continue

                        xtb_energies.append(xtb_energy)

                taut_free_energy = conformer_ensemble_free_energy(
                    xtb_energies,
                    temperature=temperature,
                )

                if taut_free_energy is None:
                    continue

                if best_free_energy is None or taut_free_energy < best_free_energy:

                    best_free_energy = taut_free_energy
                    best_taut = entry["taut"]

            except Exception as e:
                print(f"xTB failed for tautomer {entry['idx']}: {e}")

        if best_taut is None:
            return Chem.MolToSmiles(ranked[0]["taut"])
        return Chem.MolToSmiles(best_taut)
    else:
        return Chem.MolToSmiles(ranked[0]["taut"])
