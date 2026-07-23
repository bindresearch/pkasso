from dataclasses import replace
import inspect

import pandas as pd
import pytest
from rdkit import Chem

from pkasso.external.unipka.pka_predictor import free_energy as free_energy_module
from pkasso.external.unipka.pka_predictor.free_energy import (
    UNIPKA_BATCH_SIZE,
    UNIPKA_CONF_SIZE,
    UnipkaFreeEnergyConfig,
    _aggregate_free_energy_predictions,
    _aggregate_fold_free_energy_predictions,
    _free_energy_inference_argv,
    _mol_to_unmapped_smiles,
    _smiles_to_3d_coords,
    _suppress_unipka_extension_output,
    _use_gpu,
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
        UnipkaFreeEnergyConfig(nthreads=3),
    )

    assert captured == {
        "smi": "C",
        "conformer_count": 10,
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
    assert UnipkaFreeEnergyConfig().folds == (0,)


def test_free_energy_config_accepts_single_fold_without_trailing_comma():
    assert UnipkaFreeEnergyConfig(folds=(0)).folds == (0,)


def test_unipka_config_exposes_only_supported_keywords():
    assert list(inspect.signature(UnipkaFreeEnergyConfig).parameters) == [
        "model_dir",
        "folds",
        "nthreads",
        "gpu",
    ]
    with pytest.raises(TypeError):
        UnipkaFreeEnergyConfig(batch_size=4)
    with pytest.raises(TypeError):
        UnipkaFreeEnergyConfig(conf_size=3)
    with pytest.raises(TypeError):
        UnipkaFreeEnergyConfig(target_mean=6.0)
    with pytest.raises(TypeError):
        UnipkaFreeEnergyConfig(fp16=True)


def test_unipka_inference_constants_are_fixed():
    assert UNIPKA_BATCH_SIZE == 16
    assert UNIPKA_CONF_SIZE == 11


def test_gpu_defaults_to_cuda_availability(monkeypatch):
    monkeypatch.setattr(free_energy_module.torch.cuda, "is_available", lambda: True)
    assert _use_gpu(UnipkaFreeEnergyConfig()) is True

    monkeypatch.setattr(free_energy_module.torch.cuda, "is_available", lambda: False)
    assert _use_gpu(UnipkaFreeEnergyConfig()) is False


def test_gpu_true_requires_cuda(monkeypatch):
    monkeypatch.setattr(free_energy_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="gpu=True"):
        _use_gpu(UnipkaFreeEnergyConfig(gpu=True))


def test_predict_standard_free_energies_runs_only_configured_default_fold(monkeypatch, tmp_path):
    loaded_folds = []
    predicted_folds = []

    class FakeRunner:
        def __init__(self, fold):
            self.fold = fold

        def predict(self, processed_dir, task_name, n_molecules):
            predicted_folds.append(self.fold)
            return pd.DataFrame(
                {
                    "molecule_index": [0],
                    "smiles": ["CCO"],
                    "standard_free_energy": [float(self.fold)],
                    "standard_free_energy_std": [0.0],
                    "n_conformers": [1],
                }
            )

    def mock_load(cfg, fold, task_setup_dir):
        loaded_folds.append(fold)
        return FakeRunner(fold)

    monkeypatch.setattr(free_energy_module, "_write_free_energy_lmdb", lambda *args: None)
    monkeypatch.setattr(free_energy_module, "_copy_dictionaries", lambda *args: None)
    monkeypatch.setattr(free_energy_module._FreeEnergyFoldRunner, "load", staticmethod(mock_load))

    results = predict_standard_free_energies(
        [Chem.MolFromSmiles("CCO")],
        config=UnipkaFreeEnergyConfig(
            model_dir=tmp_path / "model",
        ),
    )

    assert loaded_folds == [0]
    assert predicted_folds == [0]
    assert results["standard_free_energy"].tolist() == [0.0]


def test_predict_standard_free_energies_deduplicates_smiles_and_reuses_runner(monkeypatch, tmp_path):
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

    config = UnipkaFreeEnergyConfig(
        model_dir=tmp_path / "model",
        folds=(0,),
    )

    results = predict_standard_free_energies(
        [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("N"),
            Chem.MolFromSmiles("C"),
        ],
        config=config,
    )
    predict_standard_free_energies(
        [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("N")],
        config=replace(config),
    )

    assert [call[0] for call in write_calls] == [("C", "N"), ("C", "N")]
    assert runner_predict_calls[0][1] == write_calls[0][1]
    assert runner_predict_calls[1][1] == write_calls[1][1]
    assert len(runner_load_calls) == 1
    assert [call[2] for call in runner_predict_calls] == [2, 2]
    assert results["molecule_index"].tolist() == [0, 1, 2]
    assert results["smiles"].tolist() == ["C", "N", "C"]
    assert results["standard_free_energy"].tolist() == [1.0, 2.0, 1.0]


def test_predict_standard_free_energies_runs_and_averages_five_cached_folds(monkeypatch, tmp_path):
    monkeypatch.setattr(free_energy_module, "_copy_dictionaries", lambda dict_dir, processed_dir: None)

    def mock_write_free_energy_lmdb(smiles, task_name, processed_dir, cfg):
        lmdb_path = processed_dir / task_name / "valid.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        lmdb_path.write_bytes(b"test")

    monkeypatch.setattr(free_energy_module, "_write_free_energy_lmdb", mock_write_free_energy_lmdb)

    runner_load_folds = []
    runner_predict_folds = []
    runner_task_names = []

    class FakeRunner:
        def __init__(self, fold):
            self.fold = fold

        def predict(self, processed_dir, task_name, n_molecules):
            runner_predict_folds.append(self.fold)
            runner_task_names.append(task_name)
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

    config = UnipkaFreeEnergyConfig(
        model_dir=tmp_path / "model",
        folds=(0, 1, 2, 3, 4),
    )

    results = predict_standard_free_energies([Chem.MolFromSmiles("C")], config=config)
    predict_standard_free_energies([Chem.MolFromSmiles("C")], config=config)

    assert runner_load_folds == [0, 1, 2, 3, 4]
    assert runner_predict_folds == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert len(set(runner_task_names[:5])) == 5
    assert all(task_name.endswith(f"_fold_{fold}") for fold, task_name in enumerate(runner_task_names[:5]))
    assert results["standard_free_energy"].tolist() == [2.0]
    assert results["n_folds"].tolist() == [5]


def test_changing_selected_folds_reuses_loaded_fold_and_isolates_lmdb_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(free_energy_module, "_copy_dictionaries", lambda *args: None)

    def mock_write_free_energy_lmdb(smiles, task_name, processed_dir, cfg):
        lmdb_path = processed_dir / task_name / "valid.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        lmdb_path.write_bytes(b"test")

    loaded_folds = []
    predicted = []

    class FakeRunner:
        def __init__(self, fold):
            self.fold = fold

        def predict(self, processed_dir, task_name, n_molecules):
            predicted.append((self.fold, processed_dir / task_name / "valid.lmdb"))
            return pd.DataFrame(
                {
                    "molecule_index": [0],
                    "smiles": ["C"],
                    "standard_free_energy": [float(self.fold)],
                    "standard_free_energy_std": [0.0],
                    "n_conformers": [11],
                }
            )

    def mock_load(cfg, fold, task_setup_dir):
        loaded_folds.append(fold)
        return FakeRunner(fold)

    monkeypatch.setattr(free_energy_module, "_write_free_energy_lmdb", mock_write_free_energy_lmdb)
    monkeypatch.setattr(free_energy_module._FreeEnergyFoldRunner, "load", staticmethod(mock_load))

    model_dir = tmp_path / "model"
    molecule = Chem.MolFromSmiles("C")
    predict_standard_free_energies(
        [molecule],
        config=UnipkaFreeEnergyConfig(model_dir=model_dir, folds=(0,)),
    )
    predict_standard_free_energies(
        [molecule],
        config=UnipkaFreeEnergyConfig(model_dir=model_dir, folds=(0, 1)),
    )

    assert loaded_folds == [0, 1]
    assert [fold for fold, _ in predicted] == [0, 0, 1]
    assert predicted[1][1] != predicted[2][1]
    assert predicted[1][1].name == predicted[2][1].name == "valid.lmdb"


def test_cached_free_energy_argv_omits_user_dir_to_avoid_reimport_guard(tmp_path):
    argv = _free_energy_inference_argv(
        tmp_path / "data",
        task_name="task",
        checkpoint=tmp_path / "checkpoint_best.pt",
        cfg=UnipkaFreeEnergyConfig(),
    )

    assert "--user-dir" not in argv
    assert argv[argv.index("--batch-size") + 1] == "16"
    assert argv[argv.index("--conf-size") + 1] == "11"


def test_suppress_unipka_extension_output_filters_known_fused_messages(capsys):
    with _suppress_unipka_extension_output():
        print("fused_multi_tensor is not installed corrected")
        print("fused_layer_norm is not installed corrected")
        print("ordinary setup message")

    captured = capsys.readouterr()

    assert "fused_multi_tensor" not in captured.out
    assert "fused_layer_norm" not in captured.out
    assert "ordinary setup message" in captured.out
