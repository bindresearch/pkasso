from __future__ import annotations

import hashlib
import io
import logging
import pickle
import sys
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
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
    _copy_dictionaries,
    _resolve,
    _resolve_checkpoint,
    _log,
)


PKASSO_DATA = Path(str(resources.files("pkasso") / "data"))
logger = logging.getLogger(__name__)
_UNIPKA_EXTENSION_NOISE = (
    "fused_multi_tensor is not installed corrected",
    "fused_rounding is not installed corrected",
    "fused_layer_norm is not installed corrected",
    "fused_rms_norm is not installed corrected",
    "fused_softmax is not installed corrected",
)


@dataclass(frozen=True)
class FreeEnergyPredictionConfig(PredictionConfig):
    """Configuration for direct Uni-pKa microstate free-energy inference."""

    model_dir: Path = PKASSO_DATA
    dict_dir: Path = PKASSO_DATA
    folds: tuple[int, ...] = (0,)
    target_mean: float = 6.457855284082695 # dwar + iupac (no overlap)
    loss_func: str = "infer_free_energy"
    valid_subset: str = "valid"
    conformer_gen_mode: str = "mmff"
    verbose: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.folds, int):
            object.__setattr__(self, "folds", (self.folds,))
        if not isinstance(self.folds, tuple):
            raise TypeError("folds must be an integer or tuple, for example 0 or (0, 1).")
        if not self.folds:
            raise ValueError("folds must contain at least one fold.")


@dataclass
class _FreeEnergyFoldRunner:
    cfg: FreeEnergyPredictionConfig
    fold: int
    args: object
    task: object
    model: object
    loss: object
    use_cuda: bool
    torch_module: object
    unicore_utils: object

    @classmethod
    def load(
        cls,
        cfg: FreeEnergyPredictionConfig,
        fold: int,
        task_setup_dir: Path,
    ) -> "_FreeEnergyFoldRunner":
        with _suppress_unipka_extension_output(cfg):
            _ensure_unipka_user_dir()

            try:
                from unicore import checkpoint_utils, options, tasks, utils as unicore_utils
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Uni-pKa inference requires the optional 'unicore' package."
                ) from exc

        checkpoint = _resolve_checkpoint(cfg.model_dir, fold)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        with _suppress_unipka_extension_output(cfg):
            args = _parse_free_energy_inference_args(
                task_setup_dir,
                task_name="_model_setup",
                checkpoint=checkpoint,
                cfg=replace(cfg, fold=fold),
            )
            args.cpu = not _use_fp16_or_cuda(cfg)
            args.fp16 = _use_fp16_or_cuda(cfg)

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
        return cls(cfg, fold, args, task, model, loss, use_cuda, torch, unicore_utils)

    def predict(
        self,
        processed_dir: Path,
        task_name: str,
        n_molecules: int,
    ) -> pd.DataFrame:
        self.args.data = str(processed_dir)
        self.args.task_name = task_name
        self.args.valid_subset = self.cfg.valid_subset

        batches = []
        for subset in str(self.args.valid_subset).split(","):
            self.task.load_dataset(subset, combine=False, epoch=1)
            dataset = self.task.dataset(subset)
            itr = self.task.get_batch_iterator(
                dataset=dataset,
                batch_size=self.args.batch_size,
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
                    sample = self.unicore_utils.move_to_cuda(sample) if self.use_cuda else sample
                    if len(sample) == 0:
                        continue
                    _, _, log_output = self.task.valid_step(sample, self.model, self.loss, test=True)
                    batches.append(log_output)

        return _free_energy_results_from_batches(batches, n_molecules, self.cfg.conf_size)


_FOLD_RUNNER_CACHE: dict[
    tuple[FreeEnergyPredictionConfig, int],
    _FreeEnergyFoldRunner,
] = {}


def _get_cached_fold_runner(
    cache_config: FreeEnergyPredictionConfig,
    runtime_config: FreeEnergyPredictionConfig,
    fold: int,
    task_setup_dir: Path,
) -> _FreeEnergyFoldRunner:
    """Return the process-cached model runner for one configuration and fold."""

    cache_key = (cache_config, fold)
    runner = _FOLD_RUNNER_CACHE.get(cache_key)
    if runner is None:
        runner = _FreeEnergyFoldRunner.load(runtime_config, fold, task_setup_dir)
        _FOLD_RUNNER_CACHE[cache_key] = runner
    return runner


def _ensure_unipka_user_dir() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import unimol.losses  # noqa: F401
    import unimol.models  # noqa: F401
    import unimol.tasks  # noqa: F401


@contextmanager
def _suppress_unipka_extension_output(cfg: FreeEnergyPredictionConfig):
    if cfg.verbose:
        yield
        return

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
    cfg: FreeEnergyPredictionConfig,
) -> object:
    from unicore import options

    argv = _free_energy_inference_argv(processed_dir, task_name, checkpoint, cfg)
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
    cfg: FreeEnergyPredictionConfig,
) -> list[str]:
    argv = [
        str(processed_dir),
        "--task-name",
        task_name,
        "--valid-subset",
        cfg.valid_subset,
        "--results-path",
        str(_resolve(cfg.results_dir or Path(f"{task_name}_results")) / f"fold_{cfg.fold}"),
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

    if _use_fp16_or_cuda(cfg):
        argv.append("--fp16")
    else:
        argv.append("--cpu")
    return argv


def _use_fp16_or_cuda(cfg: FreeEnergyPredictionConfig) -> bool:
    return torch.cuda.is_available() if cfg.fp16 is None else bool(cfg.fp16)


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
    """Predict standard free energies while caching loaded checkpoints."""

    cache_config = config or FreeEnergyPredictionConfig()
    cfg = cache_config
    if not mols:
        raise ValueError("At least one molecule is required for free-energy prediction.")
    if cfg.conf_size < 1:
        raise ValueError("conf_size must be at least 1.")
    smiles = [_mol_to_unmapped_smiles(mol) for mol in mols]
    unique_smiles, inverse_indices = _unique_smiles_with_inverse(smiles)
    task = task_name or _task_name_from_smiles(unique_smiles)

    with ExitStack() as stack:
        if cfg.processed_lmdb_dir is None or cfg.results_dir is None:
            tmpdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            cfg = replace(
                cfg,
                processed_lmdb_dir=cfg.processed_lmdb_dir or tmpdir / "data",
                results_dir=cfg.results_dir or tmpdir / "results",
            )

        processed_dir = _resolve(cfg.processed_lmdb_dir or Path(task))

        _write_free_energy_lmdb(unique_smiles, task, processed_dir, cfg)
        _copy_dictionaries(_resolve(cfg.dict_dir), processed_dir)

        fold_results = []
        folds = cfg.folds
        for fold in folds:
            runner = _get_cached_fold_runner(
                cache_config,
                replace(cfg, fold=fold),
                fold,
                processed_dir,
            )
            fold_results.append(
                runner.predict(processed_dir, task, len(unique_smiles))
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
    "FreeEnergyPredictionConfig",
    "predict_standard_free_energy",
    "predict_standard_free_energies",
]
