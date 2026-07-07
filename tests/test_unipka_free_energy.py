import pandas as pd
import pytest
from rdkit import Chem

from pkasso.external.unipka.pka_predictor import free_energy as free_energy_module
from pkasso.external.unipka.pka_predictor.free_energy import (
    FreeEnergyPredictionConfig,
    _aggregate_free_energy_predictions,
    _aggregate_fold_free_energy_predictions,
    _mol_to_unmapped_smiles,
    predict_standard_free_energies,
)


def test_mol_to_unmapped_smiles_removes_atom_maps_and_preserves_charge():
    mol = Chem.MolFromSmiles("[CH3:1][C:2](=O)[O-:3]")

    smiles = _mol_to_unmapped_smiles(mol)

    assert ":" not in smiles
    assert Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)) == -1


def test_aggregate_free_energy_predictions_averages_conformers_by_input_order():
    conformer_results = pd.DataFrame(
        [
            {"molecule_index": 0, "smiles": "CCO", "conformer_free_energy": 1.0},
            {"molecule_index": 0, "smiles": "CCO", "conformer_free_energy": 2.0},
            {"molecule_index": 1, "smiles": "CCN", "conformer_free_energy": -1.0},
            {"molecule_index": 1, "smiles": "CCN", "conformer_free_energy": 1.0},
        ]
    )

    results = _aggregate_free_energy_predictions(conformer_results, n_molecules=2)

    assert results["smiles"].tolist() == ["CCO", "CCN"]
    assert results["standard_free_energy"].tolist() == [1.5, 0.0]
    assert results["n_conformers"].tolist() == [2, 2]


def test_aggregate_fold_free_energy_predictions_averages_folds_by_input_order():
    fold_results = [
        pd.DataFrame(
            {
                "molecule_index": [0, 1],
                "smiles": ["CCO", "CCN"],
                "standard_free_energy": [1.0, -1.0],
                "standard_free_energy_std": [0.1, 0.2],
                "n_conformers": [11, 11],
            }
        ),
        pd.DataFrame(
            {
                "molecule_index": [0, 1],
                "smiles": ["CCO", "CCN"],
                "standard_free_energy": [3.0, 1.0],
                "standard_free_energy_std": [0.3, 0.4],
                "n_conformers": [11, 11],
            }
        ),
    ]

    results = _aggregate_fold_free_energy_predictions(fold_results, folds=(0, 1), n_molecules=2)

    assert results["smiles"].tolist() == ["CCO", "CCN"]
    assert results["standard_free_energy"].tolist() == [2.0, 0.0]
    assert results["standard_free_energy_std"].tolist() == pytest.approx([0.2, 0.3])
    assert results["standard_free_energy_fold_std"].tolist() == pytest.approx([2**0.5, 2**0.5])
    assert results["n_folds"].tolist() == [2, 2]
    assert results["n_conformers"].tolist() == [11, 11]


def test_predict_standard_free_energies_runs_all_discovered_folds(monkeypatch, tmp_path):
    calls = []
    model_dir = tmp_path / "model"
    for fold in range(5):
        checkpoint = model_dir / f"fold_{fold}" / "checkpoint_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("checkpoint")

    def mock_write_free_energy_lmdb(smiles, task_name, processed_dir, cfg):
        calls.append(("write", tuple(smiles), task_name, processed_dir))

    def mock_copy_dictionaries(dict_dir, processed_dir):
        calls.append(("copy", dict_dir, processed_dir))

    def mock_run_free_energy_inference(processed_dir, results_dir, task_name, cfg):
        calls.append(("run", cfg.fold, processed_dir, results_dir, task_name))

    def mock_read_free_energy_results(results_dir, fold, n_molecules, conf_size):
        calls.append(("read", fold, results_dir, n_molecules, conf_size))
        return pd.DataFrame(
            {
                "molecule_index": [0],
                "smiles": ["CCO"],
                "standard_free_energy": [float(fold)],
                "standard_free_energy_std": [0.0],
                "n_conformers": [conf_size],
            }
        )

    monkeypatch.setattr(free_energy_module, "_write_free_energy_lmdb", mock_write_free_energy_lmdb)
    monkeypatch.setattr(free_energy_module, "_copy_dictionaries", mock_copy_dictionaries)
    monkeypatch.setattr(free_energy_module, "_run_free_energy_inference", mock_run_free_energy_inference)
    monkeypatch.setattr(free_energy_module, "_read_free_energy_results", mock_read_free_energy_results)

    results = predict_standard_free_energies(
        [Chem.MolFromSmiles("CCO")],
        config=FreeEnergyPredictionConfig(
            model_dir=model_dir,
            processed_lmdb_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
        ),
    )

    run_folds = [call[1] for call in calls if call[0] == "run"]
    read_folds = [call[1] for call in calls if call[0] == "read"]
    assert run_folds == [0, 1, 2, 3, 4]
    assert read_folds == [0, 1, 2, 3, 4]
    assert results["standard_free_energy"].tolist() == [2.0]
    assert results["n_folds"].tolist() == [5]
