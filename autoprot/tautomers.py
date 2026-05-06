import os
import logging
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import re

logger = logging.getLogger(__name__)

def mol_to_xyz(mol, filename):
    conf = mol.GetConformer()
    with open(filename, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")


def run_xtb(xyz_file, workdir, optimize=False):

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    if optimize:
        cmd = ["xtb", xyz_file, "--opt", "--gfn2", "--alpb", "water"] # "--opt",
    else:
        cmd = ["xtb", xyz_file, "--gfn2", "--alpb", "water"] # "--opt",

    result = subprocess.run(cmd, env=env, cwd=workdir, capture_output=True, text=True)

    energy = None
    for line in result.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            # Extract float using regex (robust to formatting)
            match = re.search(r"[-+]?\d+\.\d+", line)
            if match:
                energy = float(match.group(0))
                break

    return energy

def prepare_3d(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def best_tautomer_smiles(smiles, max_tautomers: int = 100, xtb_optimize=False):

    mol = Chem.MolFromSmiles(smiles)

    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)

    if len(tautomers) == 1:
        return Chem.MolToSmiles(tautomers[0])
    if len(tautomers) > max_tautomers:
        # logger.info('Exceeding max tautomers, using input smiles.')
        print('Exceeding max tautomers, using input smiles.')
        return smiles
    
    best_energy = None
    best_mol = None

    for i, taut in enumerate(tautomers):
        try:
            taut3d = prepare_3d(Chem.Mol(taut))

            with tempfile.TemporaryDirectory() as tmpdir:
                xyz_path = os.path.join(tmpdir, f"mol_{i}.xyz")
                mol_to_xyz(taut3d, xyz_path)

                energy = run_xtb(xyz_path, tmpdir, optimize=xtb_optimize)

            if energy is None:
                continue

            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_mol = taut

        except Exception as e:
            logger.info(f"Skipping tautomer {i}: {e}")

    if best_mol is None:
        logger.info('Did not find good tautomer, using input smiles.')
        return smiles

    return Chem.MolToSmiles(best_mol)