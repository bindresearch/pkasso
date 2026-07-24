from __future__ import annotations

import hashlib
import io
import logging
import pickle
import shutil
import sys
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
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
    _copy_dictionaries,
    _missing_checkpoint_message,
    _resolve,
    _resolve_checkpoint,
)


PKASSO_DATA = Path(str(resources.files("pkasso") / "data"))
UNIPKA_BATCH_SIZE = 16
UNIPKA_CONF_SIZE = 11
_UNIPKA_DICT_DIR = PKASSO_DATA
_UNIPKA_NUM_WORKERS = 0
_UNIPKA_LOSS = "infer_free_energy"
_UNIPKA_HEAD_NAME = "chembl"
_UNIPKA_TASK_NUM = 1
_UNIPKA_ONLY_POLAR = -1
_UNIPKA_VALID_SUBSET = "valid"
_UNIPKA_CONFORMER_GEN_MODE = "mmff"
logger = logging.getLogger(__name__)
_UNIPKA_EXTENSION_NOISE = (
    "fused_multi_tensor is not installed corrected",
    "fused_rounding is not installed corrected",
    "fused_layer_norm is not installed corrected",
    "fused_rms_norm is not installed corrected",
    "fused_softmax is not installed corrected",
)


@dataclass(frozen=True)
class UnipkaFreeEnergyConfig:
    """User-configurable options for Uni-pKa free-energy inference."""

    model_dir: Path = PKASSO_DATA
    folds: tuple[int, ...] = (0,)
    nthreads: int = 16
    gpu: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        if isinstance(self.folds, int):
            object.__setattr__(self, "folds", (self.folds,))
        if not isinstance(self.folds, tuple):
            raise TypeError("folds must be an integer or tuple, for example 0 or (0, 1).")
        if not self.folds:
            raise ValueError("folds must contain at least one fold.")
        if self.nthreads < 0:
            raise ValueError("nthreads must be at least 0.")
        if self.gpu is not None and not isinstance(self.gpu, bool):
            raise TypeError("gpu must be True, False, or None.")


@dataclass
class _FreeEnergyFoldRunner:
    cfg: UnipkaFreeEnergyConfig
    fold: int
    args: object
    task: object
    model: object
    loss: object
    use_cuda: bool
    torch_module: object
    unicoreinfer_utils: object

    @classmethod
    def load(
        cls,
        cfg: UnipkaFreeEnergyConfig,
        fold: int,
        task_setup_dir: Path,
    ) -> "_FreeEnergyFoldRunner":
        checkpoint = _resolve_checkpoint(cfg.model_dir, fold)
        if not checkpoint.exists():
            raise FileNotFoundError(
                _missing_checkpoint_message(checkpoint, cfg.model_dir)
            )

        with _suppress_unipka_extension_output():
            try:
                from unicoreinfer import (
                    checkpoint_utils,
                    tasks,
                    utils as unicoreinfer_utils,
                )
                _ensure_unipka_user_dir()
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Uni-pKa inference dependencies are missing. Install them with "
                    "`python -m pip install 'pkasso[unipka]'`."
                ) from exc

        with _suppress_unipka_extension_output():
            args = _parse_free_energy_inference_args(
                task_setup_dir,
                task_name="_model_setup",
                checkpoint=checkpoint,
                cfg=cfg,
                fold=fold,
            )
            use_gpu = _use_gpu(cfg)
            args.cpu = not use_gpu
            args.fp16 = use_gpu

            use_cuda = bool(torch.cuda.is_available() and not args.cpu)
            if use_cuda:
                torch.cuda.set_device(args.device_id)

            logger.info("loading Uni-pKa model from %s", checkpoint)
            state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint))
            task = tasks.setup_task(args)
            model = task.build_model(args)
            model.load_state_dict(state["model"], strict=False)
            model.eval()
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()

            loss = task.build_loss(args)
            loss.eval()
        return cls(cfg, fold, args, task, model, loss, use_cuda, torch, unicoreinfer_utils)

    def predict(
        self,
        processed_dir: Path,
        task_name: str,
        n_molecules: int,
    ) -> pd.DataFrame:
        self.args.data = str(processed_dir)
        self.args.task_name = task_name
        self.args.valid_subset = _UNIPKA_VALID_SUBSET

        batches = []
        for subset in str(self.args.valid_subset).split(","):
            self.task.load_dataset(subset, combine=False, epoch=1)
            dataset = self.task.dataset(subset)
            itr = self.task.get_batch_iterator(
                dataset=dataset,
                batch_size=UNIPKA_BATCH_SIZE,
                ignore_invalid_inputs=True,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_shards=1,
                shard_id=0,
                num_workers=self.args.num_workers,
                data_buffer_size=self.args.data_buffer_size,
            ).next_epoch_itr(shuffle=False)

            with self.torch_module.no_grad():
                for sample in itr:
                    sample = self.unicoreinfer_utils.move_to_cuda(sample) if self.use_cuda else sample
                    if len(sample) == 0:
                        continue
                    _, _, log_output = self.task.valid_step(sample, self.model, self.loss, test=True)
                    batches.append(log_output)

        return _free_energy_results_from_batches(batches, n_molecules, UNIPKA_CONF_SIZE)


_FoldRunnerCacheKey = tuple[Path, bool | None, int]
_FOLD_RUNNER_CACHE: dict[_FoldRunnerCacheKey, _FreeEnergyFoldRunner] = {}


def _get_cached_fold_runner(
    config: UnipkaFreeEnergyConfig,
    fold: int,
    task_setup_dir: Path,
) -> _FreeEnergyFoldRunner:
    """Return the process-cached model runner for one configuration and fold."""

    cache_key = (_resolve(config.model_dir), config.gpu, fold)
    runner = _FOLD_RUNNER_CACHE.get(cache_key)
    if runner is None:
        runner = _FreeEnergyFoldRunner.load(config, fold, task_setup_dir)
        _FOLD_RUNNER_CACHE[cache_key] = runner
    return runner


def _task_name_for_fold(
    processed_dir: Path,
    task_name: str,
    fold: int,
    n_folds: int,
) -> str:
    """Give each concurrently retained fold dataset its own LMDB path."""

    if n_folds == 1:
        return task_name

    fold_task_name = f"{task_name}_fold_{fold}"
    source = processed_dir / task_name / f"{_UNIPKA_VALID_SUBSET}.lmdb"
    destination_dir = processed_dir / fold_task_name
    destination = destination_dir / f"{_UNIPKA_VALID_SUBSET}.lmdb"
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return fold_task_name


def _ensure_unipka_user_dir() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import unimol.losses  # noqa: F401
    import unimol.models  # noqa: F401
    import unimol.tasks  # noqa: F401


@contextmanager
def _suppress_unipka_extension_output():
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        yield

    _replay_filtered_unipka_output(stdout_buffer.getvalue(), original_stdout)
    _replay_filtered_unipka_output(stderr_buffer.getvalue(), original_stderr)


def _replay_filtered_unipka_output(output: str, stream: object) -> None:
    for line in output.splitlines(keepends=True):
        if any(noise in line for noise in _UNIPKA_EXTENSION_NOISE):
            continue
        stream.write(line)


def _parse_free_energy_inference_args(
    processed_dir: Path,
    *,
    task_name: str,
    checkpoint: Path,
    cfg: UnipkaFreeEnergyConfig,
    fold: int,
) -> object:
    from unicoreinfer import options

    argv = _free_energy_inference_argv(processed_dir, task_name, checkpoint, cfg, fold)
    parser = options.get_validation_parser()
    options.add_model_args(parser)

    try:
        return options.parse_args_and_arch(parser, input_args=argv)
    except TypeError:
        old_argv = sys.argv
        sys.argv = [str(REPO_ROOT / "unimol" / "infer.py"), *argv]
        try:
            return options.parse_args_and_arch(parser)
        finally:
            sys.argv = old_argv


def _free_energy_inference_argv(
    processed_dir: Path,
    task_name: str,
    checkpoint: Path,
    cfg: UnipkaFreeEnergyConfig,
    fold: int | None = None,
) -> list[str]:
    selected_fold = cfg.folds[0] if fold is None else fold
    argv = [
        str(processed_dir),
        "--task-name",
        task_name,
        "--valid-subset",
        _UNIPKA_VALID_SUBSET,
        "--results-path",
        str(processed_dir / "_results" / f"fold_{selected_fold}"),
        "--num-workers",
        str(_UNIPKA_NUM_WORKERS),
        "--ddp-backend=c10d",
        "--batch-size",
        str(UNIPKA_BATCH_SIZE),
        "--task",
        "mol_free_energy",
        "--loss",
        _UNIPKA_LOSS,
        "--arch",
        "unimol_pka",
        "--classification-head-name",
        _UNIPKA_HEAD_NAME,
        "--num-classes",
        str(_UNIPKA_TASK_NUM),
        "--dict-name",
        "dict.txt",
        "--charge-dict-name",
        "dict_charge.txt",
        "--conf-size",
        str(UNIPKA_CONF_SIZE),
        "--only-polar",
        str(_UNIPKA_ONLY_POLAR),
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

    if _use_gpu(cfg):
        argv.append("--fp16")
    else:
        argv.append("--cpu")
    return argv


def _use_gpu(cfg: UnipkaFreeEnergyConfig) -> bool:
    cuda_available = bool(torch.cuda.is_available())
    if cfg.gpu is True and not cuda_available:
        raise RuntimeError("gpu=True requires a CUDA-capable PyTorch installation and available GPU.")
    return cuda_available if cfg.gpu is None else cfg.gpu


def _unique_smiles_with_inverse(smiles: Sequence[str]) -> tuple[list[str], list[int]]:
    unique_smiles: list[str] = []
    unique_indices: dict[str, int] = {}
    inverse_indices: list[int] = []
    for smi in smiles:
        unique_idx = unique_indices.get(smi)
        if unique_idx is None:
            unique_idx = len(unique_smiles)
            unique_indices[smi] = unique_idx
            unique_smiles.append(smi)
        inverse_indices.append(unique_idx)
    return unique_smiles, inverse_indices


def _expand_unique_free_energy_predictions(
    unique_results: pd.DataFrame,
    smiles: Sequence[str],
    inverse_indices: Sequence[int],
) -> pd.DataFrame:
    if len(unique_results) <= max(inverse_indices, default=-1):
        raise ValueError(
            f"Expected at least {max(inverse_indices) + 1} unique predictions, got {len(unique_results)}."
        )

    rows = []
    for molecule_index, (smi, unique_idx) in enumerate(zip(smiles, inverse_indices)):
        row = unique_results.iloc[unique_idx].copy()
        row["molecule_index"] = molecule_index
        row["smiles"] = smi
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _free_energy_results_from_batches(
    batches: Sequence[dict[str, object]],
    n_molecules: int,
    conf_size: int,
) -> pd.DataFrame:
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


def predict_standard_free_energy(
    mol: Mol,
    *,
    config: UnipkaFreeEnergyConfig | None = None,
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
    config: UnipkaFreeEnergyConfig | None = None,
) -> pd.DataFrame:
    """Predict standard free energies while caching loaded checkpoints."""

    cfg = config or UnipkaFreeEnergyConfig()
    if not mols:
        raise ValueError("At least one molecule is required for free-energy prediction.")
    smiles = [_mol_to_unmapped_smiles(mol) for mol in mols]
    unique_smiles, inverse_indices = _unique_smiles_with_inverse(smiles)
    task = task_name or _task_name_from_smiles(unique_smiles)

    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = Path(tmpdir) / "data"

        _write_free_energy_lmdb(unique_smiles, task, processed_dir, cfg)
        _copy_dictionaries(_resolve(_UNIPKA_DICT_DIR), processed_dir)

        fold_results = []
        folds = cfg.folds
        for fold in folds:
            fold_task = _task_name_for_fold(
                processed_dir,
                task,
                fold,
                len(folds),
            )
            runner = _get_cached_fold_runner(
                cfg,
                fold,
                processed_dir,
            )
            fold_results.append(
                runner.predict(processed_dir, fold_task, len(unique_smiles))
            )

        if len(fold_results) == 1:
            unique_results = fold_results[0]
        else:
            unique_results = _aggregate_fold_free_energy_predictions(
                fold_results,
                folds,
                len(unique_smiles),
            )
        return _expand_unique_free_energy_predictions(
            unique_results,
            smiles,
            inverse_indices,
        )


def _write_free_energy_lmdb(
    smiles: Sequence[str],
    task_name: str,
    processed_dir: Path,
    cfg: UnipkaFreeEnergyConfig,
) -> None:
    """Write the LMDB layout expected by ``mol_free_energy``."""

    lmdb_path = processed_dir / task_name / f"{_UNIPKA_VALID_SUBSET}.lmdb"
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Uni-pKa free-energy inference requires the optional 'lmdb' package."
        ) from exc

    if lmdb_path.exists():
        lmdb_path.unlink()
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Preprocessing %d molecule(s) -> %s", len(smiles), lmdb_path)
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


def _smiles_to_free_energy_record(smi: str, cfg: UnipkaFreeEnergyConfig) -> dict[str, object]:
    metadata = _smiles_to_metadata(
        smi,
        conformer_count=UNIPKA_CONF_SIZE - 1,
        gen_mode=_UNIPKA_CONFORMER_GEN_MODE,
        num_threads=cfg.nthreads,
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


def _smiles_to_metadata(
    smi: str,
    conformer_count: int,
    gen_mode: str,
    num_threads: int = 1,
) -> dict[str, object]:
    if gen_mode not in {"mmff", "no_mmff"}:
        raise ValueError("conformer_gen_mode must be 'mmff' or 'no_mmff'.")

    scaffold = _smiles_to_scaffold(smi)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smi}")

    if len(mol.GetAtoms()) > 400:
        coordinates = [_smiles_to_2d_coords(smi)] * (conformer_count + 1)
    else:
        coordinates = _smiles_to_3d_coords(
            smi,
            conformer_count,
            gen_mode,
            num_threads=num_threads,
        )
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


def _smiles_to_3d_coords(
    smi: str,
    conformer_count: int,
    gen_mode: str,
    num_threads: int = 1,
) -> list[np.ndarray]:
    if conformer_count == 0:
        return []

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smi}")
    mol = AllChem.AddHs(mol)

    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=conformer_count,
                randomSeed=0,
                numThreads=num_threads,
            )
        )
        if gen_mode == "mmff":
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads)
    except Exception:
        conf_ids = []

    coordinates = [
        mol.GetConformer(int(conf_id)).GetPositions().astype(np.float32)
        for conf_id in conf_ids
    ]
    coordinates.extend(
        _smiles_to_2d_coords(smi)
        for _ in range(conformer_count - len(coordinates))
    )
    return coordinates


def _task_name_from_smiles(smiles: Sequence[str]) -> str:
    digest = hashlib.sha1("\n".join(smiles).encode("utf-8")).hexdigest()[:10]
    return f"free_energy_{digest}"


__all__ = [
    "UnipkaFreeEnergyConfig",
    "predict_standard_free_energy",
    "predict_standard_free_energies",
]
