import importlib.util
import math
import os
from pathlib import Path


def load_tautomers_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "tautomers",
        root / "autoprot" / "tautomers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rdkit_score_prior_keeps_amide_over_imidic_acid(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        if "taut_0" in xyz_file:
            return -1.0
        return 0.0

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    smiles = "O=C(NC(=O)c1ccccc1)c1ccccc1"

    assert (
        tautomers.best_tautomer_smiles(
            smiles,
            num_confs=1,
            rdkit_score_window=0,
        )
        == "O=C(NC(=O)c1ccccc1)c1ccccc1"
    )


def test_rdkit_score_prior_can_be_disabled(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        if "taut_0" in xyz_file:
            return -1.0
        return 0.0

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    smiles = "O=C(NC(=O)c1ccccc1)c1ccccc1"

    assert (
        tautomers.best_tautomer_smiles(
            smiles,
            num_confs=1,
            rdkit_score_window=None,
        )
        == "O=C(N=C(O)c1ccccc1)c1ccccc1"
    )


def test_single_surviving_rdkit_tautomer_skips_xtb(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run when only one tautomer remains")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)c1ccccc1",
            num_confs=4,
            num_xtb_confs=3,
            rdkit_score_window=0,
        )
        == "NC(=O)c1ccccc1"
    )


def test_xtb_evaluates_lowest_mmff_conformers(monkeypatch):
    tautomers = load_tautomers_module()
    xtb_inputs = []

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        xtb_inputs.append(os.path.basename(xyz_file))
        return 0.0

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    tautomers.best_tautomer_smiles(
        "NC(=O)c1ccccc1",
        num_confs=4,
        num_xtb_confs=3,
        rdkit_score_window=None,
    )

    assert len(xtb_inputs) == 6
    assert all("_conf_" in xyz_file for xyz_file in xtb_inputs)


def test_conformer_ensemble_free_energy_includes_degeneracy():
    tautomers = load_tautomers_module()

    rt = (
        tautomers.GAS_CONSTANT_KCAL_MOL_K
        * tautomers.DEFAULT_TEMPERATURE_K
        * tautomers.HARTREE_PER_KCAL_MOL
    )

    assert math.isclose(
        tautomers.conformer_ensemble_free_energy([1.0, 1.0]),
        1.0 - rt * math.log(2.0),
    )


def test_tautomer_ranking_uses_conformer_ensemble_free_energy(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_prepare_tautomer_conformers(mol, num_confs=4):
        smiles = tautomers.Chem.MolToSmiles(mol)
        if smiles == "N=C(O)c1ccccc1":
            return mol, [(0, 0.0)]
        return mol, [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]

    def mock_mol_to_xyz(mol, xyz_path, conf_id=0):
        return None

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        if "taut_0" in xyz_file:
            return 0.0
        return 0.0001

    monkeypatch.setattr(
        tautomers,
        "prepare_tautomer_conformers",
        mock_prepare_tautomer_conformers,
    )
    monkeypatch.setattr(tautomers, "mol_to_xyz", mock_mol_to_xyz)
    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)c1ccccc1",
            num_xtb_confs=4,
            rdkit_score_window=None,
        )
        == "NC(=O)c1ccccc1"
    )
