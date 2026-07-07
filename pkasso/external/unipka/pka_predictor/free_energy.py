from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from .api import (
    REPO_ROOT,
    PredictionConfig,
    _CPUUnpickler,
    _copy_dictionaries,
    _resolve,
    _resolve_checkpoint,
    _log,
)


PKASSO_DATA = Path(str(resources.files("pkasso") / "data"))


@dataclass(frozen=True)
class FreeEnergyPredictionConfig(PredictionConfig):
    """Configuration for direct Uni-pKa microstate free-energy inference."""

    model_dir: Path = PKASSO_DATA
    dict_dir: Path = PKASSO_DATA
    folds: tuple[int, ...] | None = None
    target_mean: float = 6.497260103383458
    loss_func: str = "infer_free_energy"
    valid_subset: str = "valid"
    conformer_gen_mode: str = "mmff"
    verbose: bool = False


def predict_standard_free_energy(
    mol: Mol,
    *,
    config: FreeEnergyPredictionConfig | None = None,
) -> float:
    """Predict the standard microstate formation free energy for one molecule.

    The returned value is the Uni-pKa model head output, i.e. the
    dimensionless beta-scaled standard free energy used by the FE2pKa module
    before adding the pH-dependent ``m * ln(10) * pH`` term.
    """

    results = predict_standard_free_energies([mol], config=config)
    if len(results) != 1:
        raise ValueError(f"Expected one free-energy prediction, got {len(results)}.")
    return float(results.loc[0, "standard_free_energy"])


def predict_standard_free_energies(
    mols: Sequence[Mol],
    *,
    task_name: str | None = None,
    config: FreeEnergyPredictionConfig | None = None,
) -> pd.DataFrame:
    """Predict conformer-averaged standard free energies for RDKit molecules."""

    cfg = config or FreeEnergyPredictionConfig()
    if not mols:
        raise ValueError("At least one molecule is required for free-energy prediction.")
    if cfg.conf_size < 1:
        raise ValueError("conf_size must be at least 1.")
    smiles = [_mol_to_unmapped_smiles(mol) for mol in mols]
    task = task_name or _task_name_from_smiles(smiles)

    with ExitStack() as stack:
        if cfg.processed_lmdb_dir is None or cfg.results_dir is None:
            tmpdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            cfg = replace(
                cfg,
                processed_lmdb_dir=cfg.processed_lmdb_dir or tmpdir / "data",
                results_dir=cfg.results_dir or tmpdir / "results",
            )

        processed_dir = _resolve(cfg.processed_lmdb_dir or Path(task))
        results_dir = _resolve(cfg.results_dir or Path(f"{task}_results"))

        _write_free_energy_lmdb(smiles, task, processed_dir, cfg)
        _copy_dictionaries(_resolve(cfg.dict_dir), processed_dir)

        fold_results = []
        folds = _prediction_folds(cfg)
        for fold in folds:
            fold_cfg = replace(cfg, fold=fold)
            _run_free_energy_inference(processed_dir, results_dir, task, fold_cfg)
            fold_results.append(
                _read_free_energy_results(results_dir, fold, len(smiles), cfg.conf_size)
            )

        if len(fold_results) == 1:
            return fold_results[0]
        return _aggregate_fold_free_energy_predictions(fold_results, folds, len(smiles))


def _write_free_energy_lmdb(
    smiles: Sequence[str],
    task_name: str,
    processed_dir: Path,
    cfg: FreeEnergyPredictionConfig,
) -> None:
    """Write the LMDB layout expected by ``mol_free_energy``."""

    lmdb_path = processed_dir / task_name / f"{cfg.valid_subset}.lmdb"
    if lmdb_path.exists() and not cfg.overwrite_lmdb:
        _log(cfg, f"Using existing LMDB: {lmdb_path}")
        return

    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Uni-pKa free-energy inference requires the optional 'lmdb' package."
        ) from exc

    if lmdb_path.exists():
        lmdb_path.unlink()
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    _log(cfg, f"Preprocessing {len(smiles)} molecule(s) -> {lmdb_path}")
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    try:
        with env.begin(write=True) as txn:
            for idx, smi in enumerate(smiles):
                record = _smiles_to_free_energy_record(smi, cfg)
                txn.put(str(idx).encode("ascii"), pickle.dumps(record, protocol=-1))
    finally:
        env.close()


def _smiles_to_free_energy_record(smi: str, cfg: FreeEnergyPredictionConfig) -> dict[str, object]:
    metadata = _smiles_to_metadata(
        smi,
        conformer_count=max(cfg.conf_size - 1, 0),
        gen_mode=cfg.conformer_gen_mode,
    )
    return {
        "atoms": metadata["atoms"],
        "charges": metadata["charges"],
        "coordinates": metadata["coordinates"],
        "mol": metadata["mol"],
        "smi": metadata["smi"],
        "scaffold": metadata["scaffold"],
        "target": -1.0,
    }


def _run_free_energy_inference(
    processed_dir: Path,
    results_dir: Path,
    task_name: str,
    cfg: FreeEnergyPredictionConfig,
) -> None:
    checkpoint = _resolve_checkpoint(cfg.model_dir, cfg.fold)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    fold_results_dir = results_dir / f"fold_{cfg.fold}"
    cmd = [
        cfg.python_executable,
        str(REPO_ROOT / "unimol" / "infer.py"),
        "--user-dir",
        str(REPO_ROOT / "unimol"),
        str(processed_dir),
        "--task-name",
        task_name,
        "--valid-subset",
        cfg.valid_subset,
        "--results-path",
        str(fold_results_dir),
        "--num-workers",
        str(cfg.num_workers),
        "--ddp-backend=c10d",
        "--batch-size",
        str(cfg.batch_size),
        "--task",
        "mol_free_energy",
        "--loss",
        cfg.loss_func,
        "--arch",
        "unimol_pka",
        "--classification-head-name",
        cfg.head_name,
        "--num-classes",
        str(cfg.task_num),
        "--dict-name",
        "dict.txt",
        "--charge-dict-name",
        "dict_charge.txt",
        "--conf-size",
        str(cfg.conf_size),
        "--only-polar",
        str(cfg.only_polar),
        "--path",
        str(checkpoint),
        "--fp16-init-scale",
        "4",
        "--fp16-scale-window",
        "256",
        "--log-interval",
        "50",
        "--log-format",
        "simple",
        "--required-batch-size-multiple",
        "1",
    ]

    use_fp16 = torch.cuda.is_available() if cfg.fp16 is None else cfg.fp16
    if use_fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--cpu")

    _log(cfg, f"Running standard free-energy inference with fold_{cfg.fold}")
    _run_inference_subprocess(cmd, cfg)


def _run_inference_subprocess(cmd: list[str], cfg: FreeEnergyPredictionConfig) -> None:
    if cfg.verbose:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        completed.check_returncode()


def _read_free_energy_results(
    results_dir: Path,
    fold: int,
    n_molecules: int,
    conf_size: int,
) -> pd.DataFrame:
    fold_dir = results_dir / f"fold_{fold}"
    pkl_files = sorted(fold_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No inference pickle found in {fold_dir}")

    with pkl_files[0].open("rb") as handle:
        batches = _CPUUnpickler(handle).load()

    rows = []
    row_idx = 0
    for batch in batches:
        predicts = batch["predict"].view(-1).cpu().numpy()
        for idx in range(batch["bsz"]):
            molecule_idx = row_idx // conf_size
            rows.append(
                {
                    "molecule_index": molecule_idx,
                    "conformer_index": row_idx % conf_size,
                    "smiles": batch["smi_name"][idx],
                    "conformer_free_energy": float(predicts[idx]),
                }
            )
            row_idx += 1

    conformer_results = pd.DataFrame(rows)
    return _aggregate_free_energy_predictions(conformer_results, n_molecules)


def _aggregate_free_energy_predictions(conformer_results: pd.DataFrame, n_molecules: int) -> pd.DataFrame:
    grouped = conformer_results.groupby("molecule_index", sort=True)
    results = grouped.agg(
        smiles=("smiles", "first"),
        standard_free_energy=("conformer_free_energy", "mean"),
        standard_free_energy_std=("conformer_free_energy", "std"),
        n_conformers=("conformer_free_energy", "size"),
    ).reset_index(drop=False)
    results["standard_free_energy_std"] = results["standard_free_energy_std"].fillna(0.0)

    expected = list(range(n_molecules))
    observed = results["molecule_index"].tolist()
    if observed != expected:
        raise ValueError(f"Expected molecule indices {expected}, got {observed}.")
    return results


def _prediction_folds(cfg: FreeEnergyPredictionConfig) -> tuple[int, ...]:
    if cfg.folds is None:
        return _discover_prediction_folds(cfg)

    folds = tuple(int(fold) for fold in cfg.folds)
    if not folds:
        raise ValueError("folds must contain at least one fold.")
    return folds


def _discover_prediction_folds(cfg: FreeEnergyPredictionConfig) -> tuple[int, ...]:
    model_path = _resolve(cfg.model_dir)
    if model_path.suffix == ".pt" or (model_path / "checkpoint_best.pt").exists():
        return (cfg.fold,)

    folds = []
    for checkpoint in model_path.glob("fold_*/checkpoint_best.pt"):
        fold_name = checkpoint.parent.name
        try:
            folds.append(int(fold_name.removeprefix("fold_")))
        except ValueError:
            continue

    if folds:
        return tuple(sorted(folds))
    return (cfg.fold,)


def _aggregate_fold_free_energy_predictions(
    fold_results: Sequence[pd.DataFrame],
    folds: Sequence[int],
    n_molecules: int,
) -> pd.DataFrame:
    if not fold_results:
        raise ValueError("At least one fold result is required.")
    if len(fold_results) != len(folds):
        raise ValueError(f"Expected {len(folds)} fold result(s), got {len(fold_results)}.")

    expected_indices = list(range(n_molecules))
    expected_smiles = fold_results[0]["smiles"].tolist()

    rows = []
    for fold, result in zip(folds, fold_results):
        molecule_indices = result["molecule_index"].tolist()
        if molecule_indices != expected_indices:
            raise ValueError(f"Expected molecule indices {expected_indices}, got {molecule_indices}.")
        smiles = result["smiles"].tolist()
        if smiles != expected_smiles:
            raise ValueError(f"Fold {fold} returned SMILES {smiles}, expected {expected_smiles}.")

        fold_result = result.copy()
        fold_result["fold"] = fold
        rows.append(fold_result)

    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby("molecule_index", sort=True)
    results = grouped.agg(
        smiles=("smiles", "first"),
        standard_free_energy=("standard_free_energy", "mean"),
        standard_free_energy_fold_std=("standard_free_energy", "std"),
        n_folds=("standard_free_energy", "size"),
        standard_free_energy_std=("standard_free_energy_std", "mean"),
        n_conformers=("n_conformers", "first"),
    ).reset_index(drop=False)
    results["standard_free_energy_fold_std"] = results["standard_free_energy_fold_std"].fillna(0.0)

    return results


def _mol_to_unmapped_smiles(mol: Mol) -> str:
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetAtomMapNum(0)
    mol_copy = Chem.RemoveHs(mol_copy)
    smiles = Chem.MolToSmiles(mol_copy, isomericSmiles=True)
    if not smiles:
        raise ValueError("Could not convert RDKit molecule to SMILES.")
    return smiles


def _smiles_to_metadata(smi: str, conformer_count: int, gen_mode: str) -> dict[str, object]:
    if gen_mode not in {"mmff", "no_mmff"}:
        raise ValueError("conformer_gen_mode must be 'mmff' or 'no_mmff'.")

    scaffold = _smiles_to_scaffold(smi)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smi}")

    if len(mol.GetAtoms()) > 400:
        coordinates = [_smiles_to_2d_coords(smi)] * (conformer_count + 1)
    else:
        coordinates = _smiles_to_3d_coords(smi, conformer_count, gen_mode)
        coordinates.append(_smiles_to_2d_coords(smi))

    mol_h = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol_h.GetAtoms()]

    return {
        "atoms": atoms,
        "charges": charges,
        "coordinates": coordinates,
        "mol": mol_h,
        "smi": smi,
        "scaffold": scaffold,
    }


def _smiles_to_scaffold(smi: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=True)
    except Exception:
        return smi


def _smiles_to_2d_coords(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smi}")
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    return mol.GetConformer().GetPositions().astype(np.float32)


def _smiles_to_3d_coords(smi: str, conformer_count: int, gen_mode: str) -> list[np.ndarray]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smi}")
    mol = AllChem.AddHs(mol)

    coordinates = []
    for seed in range(conformer_count):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res != 0:
                mol_tmp = Chem.MolFromSmiles(smi)
                if mol_tmp is None:
                    raise ValueError(f"Could not parse SMILES: {smi}")
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                coordinates.append(_optimize_or_get_coords(mol_tmp, smi, gen_mode))
            else:
                coordinates.append(_optimize_or_get_coords(mol, smi, gen_mode))
        except Exception:
            coordinates.append(_smiles_to_2d_coords(smi))

    return coordinates


def _optimize_or_get_coords(mol: Mol, smi: str, gen_mode: str) -> np.ndarray:
    try:
        if gen_mode == "mmff":
            AllChem.MMFFOptimizeMolecule(mol)
        coordinates = mol.GetConformer().GetPositions()
    except Exception:
        coordinates = _smiles_to_2d_coords(smi)
    return coordinates.astype(np.float32)


def _task_name_from_smiles(smiles: Sequence[str]) -> str:
    digest = hashlib.sha1("\n".join(smiles).encode("utf-8")).hexdigest()[:10]
    return f"free_energy_{digest}"


__all__ = [
    "FreeEnergyPredictionConfig",
    "predict_standard_free_energy",
    "predict_standard_free_energies",
]
