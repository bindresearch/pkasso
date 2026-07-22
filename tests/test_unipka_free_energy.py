import pandas as pd
import pytest
from rdkit import Chem

from pkasso.external.unipka.pka_predictor import free_energy as free_energy_module
from pkasso.external.unipka.pka_predictor.free_energy import (
    FreeEnergyInferenceSession,
    FreeEnergyPredictionConfig,
    _aggregate_free_energy_predictions,
    _aggregate_fold_free_energy_predictions,
    _free_energy_inference_argv,
    _run_inference_subprocess,
    _mol_to_unmapped_smiles,
    _smiles_to_3d_coords,
    _suppress_unipka_extension_output,
    predict_standard_free_energies,
)


def test_smiles_to_3d_coords_embeds_and_optimizes_conformers_together(monkeypatch):
    embed_multiple_confs = free_energy_module.AllChem.EmbedMultipleConfs
    optimize_molecule_confs = free_energy_module.AllChem.MMFFOptimizeMoleculeConfs
    calls = []

    def wrapped_embed(mol, **kwargs):
        calls.append(("embed", kwargs))
        return embed_multiple_confs(mol, **kwargs)

    def wrapped_optimize(mol, **kwargs):
        calls.append(("optimize", kwargs))
        return optimize_molecule_confs(mol, **kwargs)

    monkeypatch.setattr(free_energy_module.AllChem, "EmbedMultipleConfs", wrapped_embed)
    monkeypatch.setattr(free_energy_module.AllChem, "MMFFOptimizeMoleculeConfs", wrapped_optimize)

    coordinates = _smiles_to_3d_coords("CCO", 3, "mmff", num_threads=2)

    assert len(coordinates) == 3
    assert calls == [
        ("embed", {"numConfs": 3, "randomSeed": 0, "numThreads": 2}),
        ("optimize", {"numThreads": 2}),
    ]
    assert all(coords.dtype.name == "float32" for coords in coordinates)


def test_smiles_to_3d_coords_skips_bulk_optimization_in_no_mmff_mode(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("MMFF optimization should not run")

    monkeypatch.setattr(free_energy_module.AllChem, "MMFFOptimizeMoleculeConfs", fail_if_called)

    coordinates = _smiles_to_3d_coords("CCO", 2, "no_mmff", num_threads=2)

    assert len(coordinates) == 2


def test_smiles_to_3d_coords_falls_back_to_2d_when_bulk_embedding_fails(monkeypatch):
    def fail_embedding(*args, **kwargs):
        raise ValueError("embedding failed")

    monkeypatch.setattr(free_energy_module.AllChem, "EmbedMultipleConfs", fail_embedding)

    coordinates = _smiles_to_3d_coords("CCO", 2, "mmff", num_threads=2)

    assert len(coordinates) == 2
    assert all(coords.dtype.name == "float32" for coords in coordinates)


def test_free_energy_record_passes_configured_threads_to_conformer_generation(monkeypatch):
    captured = {}

    def mock_metadata(smi, conformer_count, gen_mode, num_threads):
        captured.update(
            smi=smi,
            conformer_count=conformer_count,
            gen_mode=gen_mode,
            num_threads=num_threads,
        )
        return {
            "atoms": ["C"],
            "charges": [0],
            "coordinates": [],
            "mol": Chem.MolFromSmiles("C"),
            "smi": smi,
            "scaffold": "",
        }

    monkeypatch.setattr(free_energy_module, "_smiles_to_metadata", mock_metadata)

    free_energy_module._smiles_to_free_energy_record(
        "C",
        FreeEnergyPredictionConfig(conf_size=5, nthreads=3),
    )

    assert captured == {
        "smi": "C",
        "conformer_count": 4,
        "gen_mode": "mmff",
        "num_threads": 3,
    }


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


def test_free_energy_config_defaults_to_one_fold():
    assert FreeEnergyPredictionConfig().folds == (0,)


def test_free_energy_config_accepts_single_fold_without_trailing_comma():
    assert FreeEnergyPredictionConfig(folds=(0)).folds == (0,)


def test_predict_standard_free_energies_runs_only_configured_default_fold(monkeypatch, tmp_path):
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
    assert run_folds == [0]
    assert read_folds == [0]
    assert results["standard_free_energy"].tolist() == [0.0]


def test_free_energy_inference_session_deduplicates_smiles_and_reuses_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(free_energy_module, "_copy_dictionaries", lambda dict_dir, processed_dir: None)

    write_calls = []
    runner_load_calls = []
    runner_predict_calls = []

    def mock_write_free_energy_lmdb(smiles, task_name, processed_dir, cfg):
        write_calls.append((tuple(smiles), task_name, processed_dir))

    class FakeRunner:
        def predict(self, processed_dir, task_name, n_molecules):
            runner_predict_calls.append((processed_dir, task_name, n_molecules))
            return pd.DataFrame(
                {
                    "molecule_index": [0, 1],
                    "smiles": ["C", "N"],
                    "standard_free_energy": [1.0, 2.0],
                    "standard_free_energy_std": [0.0, 0.0],
                    "n_conformers": [1, 1],
                }
            )

    def mock_load(cfg, fold, task_setup_dir):
        runner_load_calls.append((cfg, fold, task_setup_dir))
        return FakeRunner()

    monkeypatch.setattr(free_energy_module, "_write_free_energy_lmdb", mock_write_free_energy_lmdb)
    monkeypatch.setattr(free_energy_module._FreeEnergyFoldRunner, "load", staticmethod(mock_load))

    session = FreeEnergyInferenceSession(
        FreeEnergyPredictionConfig(
            model_dir=tmp_path / "model",
            dict_dir=tmp_path / "dict",
            processed_lmdb_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            folds=(0,),
            conf_size=1,
        )
    )

    results = session.predict_standard_free_energies(
        [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("N"),
            Chem.MolFromSmiles("C"),
        ]
    )
    session.predict_standard_free_energies([Chem.MolFromSmiles("C"), Chem.MolFromSmiles("N")])

    assert [call[0] for call in write_calls] == [("C", "N"), ("C", "N")]
    assert write_calls[0][1] != write_calls[1][1]
    assert runner_predict_calls[0][1] == write_calls[0][1]
    assert runner_predict_calls[1][1] == write_calls[1][1]
    assert len(runner_load_calls) == 1
    assert [call[2] for call in runner_predict_calls] == [2, 2]
    assert results["molecule_index"].tolist() == [0, 1, 2]
    assert results["smiles"].tolist() == ["C", "N", "C"]
    assert results["standard_free_energy"].tolist() == [1.0, 2.0, 1.0]


def test_free_energy_inference_session_runs_and_averages_five_folds(monkeypatch, tmp_path):
    monkeypatch.setattr(free_energy_module, "_copy_dictionaries", lambda dict_dir, processed_dir: None)

    def mock_write_free_energy_lmdb(smiles, task_name, processed_dir, cfg):
        lmdb_path = processed_dir / task_name / f"{cfg.valid_subset}.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        lmdb_path.write_bytes(b"")

    monkeypatch.setattr(free_energy_module, "_write_free_energy_lmdb", mock_write_free_energy_lmdb)

    runner_load_folds = []
    runner_predict_folds = []

    class FakeRunner:
        def __init__(self, fold):
            self.fold = fold

        def predict(self, processed_dir, task_name, n_molecules):
            runner_predict_folds.append(self.fold)
            return pd.DataFrame(
                {
                    "molecule_index": [0],
                    "smiles": ["C"],
                    "standard_free_energy": [float(self.fold)],
                    "standard_free_energy_std": [0.0],
                    "n_conformers": [1],
                }
            )

    def mock_load(cfg, fold, task_setup_dir):
        runner_load_folds.append(fold)
        return FakeRunner(fold)

    monkeypatch.setattr(free_energy_module._FreeEnergyFoldRunner, "load", staticmethod(mock_load))

    session = FreeEnergyInferenceSession(
        FreeEnergyPredictionConfig(
            model_dir=tmp_path / "model",
            dict_dir=tmp_path / "dict",
            processed_lmdb_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            folds=(0, 1, 2, 3, 4),
            conf_size=1,
        )
    )

    results = session.predict_standard_free_energies([Chem.MolFromSmiles("C")])
    session.predict_standard_free_energies([Chem.MolFromSmiles("C")])

    assert runner_load_folds == [0, 1, 2, 3, 4]
    assert runner_predict_folds == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert results["standard_free_energy"].tolist() == [2.0]
    assert results["n_folds"].tolist() == [5]


def test_reusable_free_energy_argv_omits_user_dir_to_avoid_reimport_guard(tmp_path):
    argv = _free_energy_inference_argv(
        tmp_path / "data",
        task_name="task",
        checkpoint=tmp_path / "checkpoint_best.pt",
        cfg=FreeEnergyPredictionConfig(results_dir=tmp_path / "results"),
    )

    assert "--user-dir" not in argv


def test_suppress_unipka_extension_output_filters_known_fused_messages(capsys):
    with _suppress_unipka_extension_output(FreeEnergyPredictionConfig()):
        print("fused_multi_tensor is not installed corrected")
        print("fused_layer_norm is not installed corrected")
        print("ordinary setup message")

    captured = capsys.readouterr()

    assert "fused_multi_tensor" not in captured.out
    assert "fused_layer_norm" not in captured.out
    assert "ordinary setup message" in captured.out


def test_suppress_unipka_extension_output_respects_verbose(capsys):
    with _suppress_unipka_extension_output(FreeEnergyPredictionConfig(verbose=True)):
        print("fused_multi_tensor is not installed corrected")

    captured = capsys.readouterr()

    assert "fused_multi_tensor is not installed corrected" in captured.out


def test_run_inference_subprocess_is_quiet_by_default(monkeypatch):
    captured = {}

    class CompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def mock_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return CompletedProcess()

    monkeypatch.setattr(free_energy_module.subprocess, "run", mock_run)

    _run_inference_subprocess(["python", "infer.py"], FreeEnergyPredictionConfig())

    assert captured["cmd"] == ["python", "infer.py"]
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True


def test_run_inference_subprocess_respects_verbose(monkeypatch):
    captured = {}

    def mock_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(free_energy_module.subprocess, "run", mock_run)

    _run_inference_subprocess(["python", "infer.py"], FreeEnergyPredictionConfig(verbose=True))

    assert captured["cmd"] == ["python", "infer.py"]
    assert captured["kwargs"]["check"] is True
    assert "capture_output" not in captured["kwargs"]
