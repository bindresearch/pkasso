import importlib.util
from pathlib import Path


def load_tautomers_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "tautomers",
        root / "pkasso" / "tautomers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rdkit_score_prior_keeps_amide_over_imidic_acid():
    tautomers = load_tautomers_module()

    smiles = "O=C(NC(=O)c1ccccc1)c1ccccc1"

    assert (
        tautomers.best_tautomer_smiles(
            smiles,
            num_confs=1,
        )
        == "O=C(NC(=O)c1ccccc1)c1ccccc1"
    )


def test_single_rdkit_tautomer_is_returned():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)c1ccccc1",
            num_confs=4,
        )
        == "NC(=O)c1ccccc1"
    )


def test_imidic_acid_filter_prefers_amide():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)c1ccccc1",
            num_confs=1,
        )
        == "NC(=O)c1ccccc1"
    )


def test_imidic_acid_filter_handles_two_amides():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)C(N)=O",
            num_confs=1,
        )
        == "NC(=O)C(N)=O"
    )


def test_thioimidic_acid_filter_prefers_thioamide():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=S)c1ccccc1",
            num_confs=1,
        )
        == "NC(=S)c1ccccc1"
    )


def test_thioimidic_acid_filter_handles_two_thioamides():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=S)C(N)=S",
            num_confs=1,
        )
        == "NC(=S)C(N)=S"
    )


def test_hydroxamate_filter_prefers_carbonyl_over_hydroximic_acid():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "O=C(NO)C1c2ccccc2Oc2ccccc21",
            num_confs=1,
        )
        == "O=C(NO)C1c2ccccc2Oc2ccccc21"
    )


def test_hydroxamate_filter_keeps_hydroxamate_motif():
    tautomers = load_tautomers_module()

    smiles = "CN1C(=O)N(C[C@H](C(=O)NO)[C@@H](CC2CCCC2)C(=O)N2CCCCC2)C(=O)C1(C)C"

    assert "C(=O)NO" in tautomers.best_tautomer_smiles(
        smiles,
        num_confs=1,
    )


def test_max_tautomers_falls_back_to_input_smiles(capsys):
    tautomers = load_tautomers_module()

    smiles = "O=C(NC(=O)c1ccccc1)c1ccccc1"

    assert (
        tautomers.best_tautomer_smiles(
            smiles,
            max_tautomers=1,
            num_confs=1,
        )
        == smiles
    )

    assert "Exceeding max tautomers" in capsys.readouterr().out


def test_returns_input_smiles_when_conformer_generation_fails(monkeypatch):
    tautomers = load_tautomers_module()

    monkeypatch.setattr(tautomers, "rdkit_tautomer_conformers", lambda mol, num_confs=10: None)

    smiles = "O=C(NO)C1c2ccccc2Oc2ccccc21"

    assert (
        tautomers.best_tautomer_smiles(
            smiles,
            num_confs=1,
        )
        == smiles
    )


def test_tautomer_ranking_only_prepares_tautomers_inside_score_window(monkeypatch):
    tautomers = load_tautomers_module()

    calls = []

    def mock_rdkit_tautomer_conformers(mol, num_confs=10):
        smiles = tautomers.Chem.MolToSmiles(mol)
        calls.append(smiles)
        energy = 100.0 if smiles == "O=c1cccc[nH]1" else 0.0
        return mol, [(0, energy)]

    monkeypatch.setattr(
        tautomers,
        "rdkit_tautomer_conformers",
        mock_rdkit_tautomer_conformers,
    )

    assert (
        tautomers.best_tautomer_smiles(
            "O=C1C=CC=CN1",
            num_confs=1,
        )
        == "O=c1cccc[nH]1"
    )
    assert calls == ["O=c1cccc[nH]1"]


def test_tautomer_ranking_uses_mmff_energy_within_score_window(monkeypatch):
    tautomers = load_tautomers_module()

    calls = []

    def mock_rdkit_tautomer_conformers(mol, num_confs=10):
        smiles = tautomers.Chem.MolToSmiles(mol)
        calls.append(smiles)
        energy = 0.0 if smiles == "Oc1ccccn1" else 10.0
        return mol, [(0, energy)]

    monkeypatch.setattr(
        tautomers,
        "rdkit_tautomer_conformers",
        mock_rdkit_tautomer_conformers,
    )

    assert (
        tautomers.best_tautomer_smiles(
            "O=C1C=CC=CN1",
            num_confs=1,
            score_window=2,
        )
        == "Oc1ccccn1"
    )
    assert calls == ["O=c1cccc[nH]1", "Oc1ccccn1"]


def test_tautomer_ranking_uses_mmff_energy_as_tiebreaker(monkeypatch):
    tautomers = load_tautomers_module()

    def mock_score_tautomer(self, mol):
        return 1

    def mock_rdkit_tautomer_conformers(mol, num_confs=10):
        smiles = tautomers.Chem.MolToSmiles(mol)
        energy = 0.0 if smiles == "N=C(O)c1ccccc1" else 10.0
        return mol, [(0, energy)]

    monkeypatch.setattr(
        tautomers.rdMolStandardize.TautomerEnumerator,
        "ScoreTautomer",
        mock_score_tautomer,
    )
    monkeypatch.setattr(
        tautomers,
        "rdkit_tautomer_conformers",
        mock_rdkit_tautomer_conformers,
    )

    assert (
        tautomers.best_tautomer_smiles(
            "NC(=O)c1ccccc1",
            num_confs=1,
        )
        == "N=C(O)c1ccccc1"
    )


def test_invalid_smiles_is_returned_unchanged():
    tautomers = load_tautomers_module()

    assert (
        tautomers.best_tautomer_smiles(
            "not a smiles",
            num_confs=1,
        )
        == "not a smiles"
    )
