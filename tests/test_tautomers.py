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
            use_xtb=True,
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


def test_imidic_acid_filter_overrides_wide_rdkit_window(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run after imidic acid filtering")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)c1ccccc1",
            num_confs=1,
            rdkit_score_window=5,
        )
        == "NC(=O)c1ccccc1"
    )


def test_imidic_acid_filter_handles_two_amides(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run after imidic acid filtering")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)C(N)=O",
            num_confs=1,
            rdkit_score_window=5,
        )
        == "NC(=O)C(N)=O"
    )


def test_thioimidic_acid_filter_overrides_wide_rdkit_window(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run after thioimidic acid filtering")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=S)c1ccccc1",
            num_confs=1,
            rdkit_score_window=5,
        )
        == "NC(=S)c1ccccc1"
    )


def test_thioimidic_acid_filter_handles_two_thioamides(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run after thioimidic acid filtering")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=S)C(N)=S",
            num_confs=1,
            rdkit_score_window=5,
        )
        == "NC(=S)C(N)=S"
    )


def test_hydroxamate_filter_prefers_carbonyl_over_hydroximic_acid(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run after hydroxamate filtering")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    assert (
        tautomers.best_tautomer_smiles(
            "O=C(NO)C1c2ccccc2Oc2ccccc21",
            num_confs=1,
            rdkit_score_window=1,
        )
        == "O=C(NO)C1c2ccccc2Oc2ccccc21"
    )


def test_hydroxamate_filter_runs_before_rdkit_score_window(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_run_xtb(xyz_file, workdir, optimize=False):
        raise AssertionError("xTB should not run after hydroxamate filtering")

    monkeypatch.setattr(tautomers, "run_xtb", mock_run_xtb)

    smiles = (
        "CN1C(=O)N(C[C@H](C(=O)NO)[C@@H](CC2CCCC2)"
        "C(=O)N2CCCCC2)C(=O)C1(C)C"
    )

    assert "C(=O)NO" in tautomers.best_tautomer_smiles(
        smiles,
        num_confs=1,
        rdkit_score_window=1,
    )


def test_tautomer_debug_prints_candidates(capsys):
    tautomers = load_tautomers_module()

    tautomers.best_tautomer_smiles(
        "O=C(NO)C1c2ccccc2Oc2ccccc21",
        num_confs=1,
        rdkit_score_window=1,
        debug=True,
    )

    output = capsys.readouterr().out

    assert "Tautomer candidates after MMFF preparation" in output
    assert "Tautomer candidates after tautomer filters" in output
    assert "hydroxamate" in output


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
        use_xtb=True,
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
            use_xtb=True,
            num_xtb_confs=4,
            rdkit_score_window=None,
        )
        == "NC(=O)c1ccccc1"
    )
