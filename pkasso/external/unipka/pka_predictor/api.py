from __future__ import annotations

import io
import hashlib
import pickle
import shutil
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location="cpu",
                weights_only=False,
            )
        return super().find_class(module, name)


@dataclass(frozen=True)
class PredictionConfig:
    processed_lmdb_dir: Path | None = None
    results_dir: Path | None = None
    model_dir: Path = Path("data")
    dict_dir: Path = Path("data")
    fold: int = 0
    batch_size: int = 16
    conf_size: int = 11
    nthreads: int = 16
    num_workers: int = 0
    head_name: str = "chembl"
    task_num: int = 1
    only_polar: int = -1
    loss_func: str = "finetune_mse"
    python_executable: str = sys.executable
    fp16: bool | None = None
    overwrite_lmdb: bool = True
    write_csv: bool = False
    verbose: bool = True


def predict_pkas(
    input_file: str | Path,
    *,
    task_name: str | None = None,
    config: PredictionConfig | None = None,
) -> pd.DataFrame:
    """Predict pKa values from a Uni-pKa TSV file.

    The input TSV should contain a ``SMILES`` column. If ``TARGET`` is absent,
    preprocessing will fill it with ``-1.0`` as the existing script does.
    """
    cfg = config or PredictionConfig()
    input_path = _resolve(input_file)
    task = task_name or input_path.stem
    processed_dir = _resolve(cfg.processed_lmdb_dir or Path(task))
    results_dir = _resolve(cfg.results_dir or Path(f"{task}_results"))

    _preprocess(input_path, task, processed_dir, cfg)
    _copy_dictionaries(_resolve(cfg.dict_dir), processed_dir)

    _run_inference(processed_dir, results_dir, task, cfg)
    final_results = _read_results(results_dir, cfg.fold, task)

    if cfg.write_csv:
        results_dir.mkdir(parents=True, exist_ok=True)
        final_results.to_csv(results_dir / "predictions.csv", index=False)

    return final_results


def predict_pkas_from_smiles(
    smiles_acid: str | Sequence[str],
    smiles_base: str | Sequence[str],
    *,
    target: float = -1.0,
    task_name: str | None = None,
    config: PredictionConfig | None = None,
) -> pd.DataFrame:
    """Predict pKa from acid/base macrostate SMILES directly."""
    acid = _macrostate_to_smiles(smiles_acid)
    base = _macrostate_to_smiles(smiles_base)
    reaction_smiles = f"{acid}>>{base}"
    task = task_name or _task_name_from_smiles(reaction_smiles)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"{task}.tsv"
        pd.DataFrame(
            [{"SMILES": reaction_smiles, "TARGET": target}]
        ).to_csv(input_path, index=False, sep="\t")
        return predict_pkas(input_path, task_name=task, config=config)


def _preprocess(
    input_path: Path,
    task_name: str,
    processed_dir: Path,
    cfg: PredictionConfig,
) -> None:
    lmdb_path = processed_dir / f"{task_name}.lmdb"
    if lmdb_path.exists() and not cfg.overwrite_lmdb:
        _log(cfg, f"Using existing LMDB: {lmdb_path}")
        return
    _log(cfg, f"Preprocessing {input_path} -> {lmdb_path}")
    from scripts.preprocess_pka import write_lmdb

    write_lmdb(
        task_name=task_name,
        input_csv=str(input_path),
        output_dir=str(processed_dir),
        nthreads=cfg.nthreads,
    )


def _copy_dictionaries(dict_dir: Path, processed_dir: Path) -> None:
    for name in ("dict.txt", "dict_charge.txt"):
        src = dict_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing dictionary file: {src}")
        shutil.copy2(src, processed_dir / name)


def _run_inference(
    processed_dir: Path,
    results_dir: Path,
    task_name: str,
    cfg: PredictionConfig,
) -> None:
    checkpoint = _resolve_checkpoint(cfg.model_dir, cfg.fold)
    if not checkpoint.exists():
        raise FileNotFoundError(_missing_checkpoint_message(checkpoint, cfg.model_dir))

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
        task_name,
        "--results-path",
        str(fold_results_dir),
        "--num-workers",
        str(cfg.num_workers),
        "--ddp-backend=c10d",
        "--batch-size",
        str(cfg.batch_size),
        "--task",
        "mol_pka",
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

    _log(cfg, f"Running inference with fold_{cfg.fold}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _resolve_checkpoint(model_dir: str | Path, fold: int) -> Path:
    model_path = _resolve(model_dir)
    if model_path.suffix == ".pt":
        return model_path

    checkpoint = model_path / "checkpoint_best.pt"
    if checkpoint.exists():
        return checkpoint

    return model_path / f"fold_{fold}" / "checkpoint_best.pt"


def _missing_checkpoint_message(checkpoint: Path, model_dir: str | Path) -> str:
    output_folder = shlex.quote(str(Path(model_dir)))
    return (
        f"Missing Uni-pKa checkpoint: {checkpoint}. Download the model weights with "
        f"`pkasso-download-unipka-model --output-folder {output_folder}`."
    )


def _read_results(results_dir: Path, fold: int, task_name: str) -> pd.DataFrame:
    fold_dir = results_dir / f"fold_{fold}"
    preferred = fold_dir / f"fold_{fold}_{task_name}.out.pkl"
    pkl_files = [preferred] if preferred.exists() else sorted(fold_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No inference pickle found in {fold_dir}")

    with pkl_files[0].open("rb") as handle:
        batches = _CPUUnpickler(handle).load()

    rows = []
    for batch in batches:
        for idx in range(batch["bsz"]):
            rows.append(
                {
                    "smiles": batch["smi_name"][idx],
                    "predict": batch["predict"][idx].cpu().item(),
                    "target": batch["target"][idx].cpu().item(),
                }
            )
    return pd.DataFrame(rows)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _log(cfg: PredictionConfig, message: str) -> None:
    if cfg.verbose:
        print(message)


def _macrostate_to_smiles(smiles: str | Sequence[str]) -> str:
    if isinstance(smiles, str):
        return smiles
    if not smiles:
        raise ValueError("Macrostate SMILES sequence must not be empty")
    return ",".join(smiles)


def _task_name_from_smiles(reaction_smiles: str) -> str:
    digest = hashlib.sha1(reaction_smiles.encode("utf-8")).hexdigest()[:10]
    return f"smiles_{digest}"
