"""Command-line interface for running pKasso workflows."""

from __future__ import annotations

import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import click
from joblib import Parallel, delayed
from rdkit.Chem.rdchem import Mol

COMMANDS = {"single", "batch", "scan"}


class RejectDuplicateOptionsCommand(click.Command):
    """Click command that rejects repeated occurrences of the same option."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        options_by_flag: dict[str, click.Option] = {}
        for param in self.get_params(ctx):
            if isinstance(param, click.Option):
                for flag in (*param.opts, *param.secondary_opts):
                    options_by_flag[flag] = param

        seen: set[str] = set()
        for arg in args:
            if arg == "--":
                break
            flag = arg.split("=", maxsplit=1)[0]
            option = options_by_flag.get(flag)
            if option is None:
                continue
            if option.name in seen:
                raise click.UsageError(
                    f"Option {option.get_error_hint(ctx)} cannot be specified multiple times.",
                    ctx,
                )
            seen.add(option.name)

        return super().parse_args(ctx, args)


class RejectDuplicateOptionsGroup(RejectDuplicateOptionsCommand, click.Group):
    """Click group whose subcommands reject duplicate options."""

    command_class = RejectDuplicateOptionsCommand


def _compute_protonate(
    idx: int,
    smiles: str,
    name: str = "molecule",
    **kwargs: Any,
) -> tuple[int, str, list[str], list[Mol]]:
    smiles_out, mols_out = protonate(smiles, name=name, **kwargs)
    return idx, name, smiles_out, mols_out


def protonate(*args: Any, **kwargs: Any) -> Any:
    from .py_interface import protonate as _protonate

    return _protonate(*args, **kwargs)


def scan_pH(*args: Any, **kwargs: Any) -> Any:
    from .py_interface import scan_pH as _scan_pH

    return _scan_pH(*args, **kwargs)


def read_smi(*args: Any, **kwargs: Any) -> Any:
    from .utils import read_smi as _read_smi

    return _read_smi(*args, **kwargs)


def save_sdf(*args: Any, **kwargs: Any) -> Any:
    from .postprocess import save_sdf as _save_sdf

    return _save_sdf(*args, **kwargs)


def _safe_filename_stem(name: str) -> str:
    """Return a short, filesystem-safe name."""

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return (safe_name or "molecule")[:200]


def _safe_sdf_filename(idx: int, name: str) -> str:
    """Return a unique, filesystem-safe filename for a batch molecule."""

    return f"{idx:04d}_{_safe_filename_stem(name)}.sdf"


def _check_distinct_paths(paths: Iterable[tuple[str, Path | None]]) -> None:
    """Reject input and output options resolving to the same path."""

    seen: dict[Path, str] = {}
    for label, path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            raise click.UsageError(f"{label} must not use the same path as {seen[resolved]}.")
        seen[resolved] = label


def format_microstate_table(
    name: str,
    pH: float,
    smiles: Sequence[str],
    mols: Sequence[Mol],
    name_width: int = 0,
    smiles_width: int = 0,
    output_width: int = 0,
) -> str:
    """Format pH-specific microstate output as a plain-text table."""

    max_sm = max((len(sm) for sm in smiles), default=0)
    max_name = max((len(str(mol.GetProp("_Name"))) for mol in mols), default=0)

    name_width = max(name_width, max_name, len("Microstate"))
    smiles_width = max(smiles_width, max_sm, len("SMILES"))
    title = f"{name} | pH: {pH:}"
    table_width = name_width + smiles_width + 29
    output_width = max(output_width, table_width + 2, len(title) + 2)

    lines = [
        "-" * output_width,
        title.center(output_width),
        f"{'Microstate':{name_width}s} {'SMILES':{smiles_width}s} {'Probability':>13s} {'Net charge':>13s}",
        "-" * output_width,
    ]

    for sm, mol in zip(smiles, mols):
        name_state = mol.GetProp("_Name")
        probability = float(mol.GetProp("Probability"))
        net_charge = float(mol.GetProp("net_charge"))
        lines.append(
            f"{name_state:{name_width}s} {sm:{smiles_width}s} "
            f"{probability:>13.5f} {net_charge:>13.0f}"
        )

    return "\n".join(lines)


def write_microstate_tables(
    txt_out: Path,
    outputs: Iterable[tuple[str, float, Sequence[str], Sequence[Mol]]],
) -> None:
    """Write one or more microstate tables to a common text file."""

    formatted_output = format_microstate_tables(outputs)

    with open(txt_out, "w", encoding="utf-8") as output_file:
        if formatted_output:
            output_file.write(f"{formatted_output}\n")


def format_microstate_tables(
    outputs: Iterable[tuple[str, float, Sequence[str], Sequence[Mol]]],
) -> str:
    """Format multiple microstate tables using common column widths."""

    outputs = list(outputs)
    name_width = max(
        (len(str(mol.GetProp("_Name"))) for _, _, _, mols in outputs for mol in mols),
        default=len("Microstate"),
    )
    smiles_width = max(
        (len(smiles) for _, _, smiles_all, _ in outputs for smiles in smiles_all),
        default=len("SMILES"),
    )
    table_width = max(name_width, len("Microstate")) + max(smiles_width, len("SMILES")) + 31
    title_width = max((len(f"{name} | pH: {pH:}") + 2 for name, pH, _, _ in outputs), default=0)
    output_width = max(table_width, title_width)

    return "\n\n\n".join(
        format_microstate_table(
            name,
            pH,
            smiles,
            mols,
            name_width=name_width,
            smiles_width=smiles_width,
            output_width=output_width,
        )
        for name, pH, smiles, mols in outputs
    )


def _common_option_conflicts(ctx: click.Context) -> None:
    """Raise a Click error for explicitly incompatible shared CLI options."""

    params = ctx.params
    commandline = click.core.ParameterSource.COMMANDLINE
    max_tautomers_source = ctx.get_parameter_source("max_tautomers")
    tautomer_search_source = ctx.get_parameter_source("tautomer_search")
    num_confs_source = ctx.get_parameter_source("num_confs")

    if (
        max_tautomers_source == commandline
        and tautomer_search_source == commandline
        and params.get("tautomer_search") is False
    ):
        raise click.UsageError("--max-tautomers cannot be used with --no-tautomer-search.")

    if (
        num_confs_source == commandline
        and tautomer_search_source == commandline
        and params.get("tautomer_search") is False
    ):
        raise click.UsageError("--num-confs cannot be used with --no-tautomer-search.")

    cutoff_states = params.get("cutoff_states")
    if cutoff_states is not None and cutoff_states < 1:
        raise click.UsageError("--cutoff-states must be >= 1.")

    cutoff_export = ctx.params.get("cutoff_export")
    if cutoff_export is not None and (
        not math.isfinite(cutoff_export)
        or (cutoff_export < 0)
        or (cutoff_export > 1)
    ):
        raise click.UsageError("--cutoff-export must be >= 0 and <= 1.")

    for parameter, option in (
        ("ph", "--ph"),
        ("min_ph", "--min-ph"),
        ("max_ph", "--max-ph"),
    ):
        value = params.get(parameter)
        if value is not None and not math.isfinite(value):
            raise click.UsageError(f"{option} must be finite.")

    if params.get("njobs") == 0:
        raise click.UsageError("--njobs must not be 0.")

    if params.get("model") == "molgpka":
        if params.get("gpu") and ctx.get_parameter_source("gpu") == commandline:
            raise click.UsageError("--gpu requires --model unipka or --model mixed.")
        if ctx.get_parameter_source("unipka_model_folder") == commandline:
            raise click.UsageError(
                "--unipka-model-folder requires --model unipka or --model mixed."
            )

    if params.get("individual_sdfs") is False:
        if ctx.get_parameter_source("path_out") == commandline:
            raise click.UsageError("--path-out requires --individual-sdfs.")
        if ctx.get_parameter_source("overwrite") == commandline:
            raise click.UsageError("--overwrite/--no-overwrite requires --individual-sdfs.")

    min_ph = params.get("min_ph")
    max_ph = params.get("max_ph")

    if (min_ph is not None) and (max_ph is not None):
        if min_ph > max_ph:
            raise click.UsageError("--max-ph must not be smaller than --min-ph.")

def _model_kwargs(
    model: str,
    unipka_model_folder: Path | None,
    gpu: bool,
) -> dict[str, Any]:
    """Build Python-interface arguments for the selected CLI model."""

    if model == "molgpka":
        return {}

    unipka_options: dict[str, object] = {
        "folds": 1,
        "gpu": gpu,
    }
    if unipka_model_folder is not None:
        unipka_options["model_dir"] = unipka_model_folder

    if model == "unipka":
        models = {"unipka": unipka_options}
    else:
        models = {
            "molgpka": {},
            "unipka": unipka_options,
        }

    return {"model": models}

COMMON_OPTIONS = [
    click.option(
        "--model",
        type=click.Choice(["molgpka", "unipka", "mixed"]),
        default="molgpka",
        show_default=True,
        help="pKa model to use; mixed combines MolGpKa with Uni-pKa",
    ),
    click.option(
        "--unipka-model-folder",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help=(
            "Uni-pKa model cache folder; defaults to unipkainfer's per-user "
            "model directory"
        ),
    ),
    click.option(
        "--nthreads",
        type=click.IntRange(min=0),
        default=0,
        show_default=True,
        help="Threads used by RDKit, MolGpKa, and Uni-pKa; 0 selects automatically",
    ),
    click.option(
        "--gpu/--no-gpu",
        default=False,
        show_default=True,
        help="Use a GPU for Uni-pKa inference",
    ),
    click.option(
        "--matrix-def",
        type=click.Choice(["dG", "msm"]),
        default="dG",
        show_default=True,
        help="Use free energy differences or Markov state model to determine microstate probabilities",
    ),
    click.option(
        "--cutoff-states",
        type=int,
        default=1000,
        show_default=True,
        help="Max. number of microstates per coupled cluster of protonation sites",
    ),
    click.option(
        "--tautomer-search/--no-tautomer-search",
        is_flag=True,
        default=True,
        show_default=True,
        help="Run tautomer search before pKasso.",
    ),
    click.option(
        "--max-tautomers",
        type=int,
        default=20,
        show_default=True,
        help="Max. number of tautomers to enumerate",
    ),
    click.option(
        "--num-confs",
        type=int,
        default=10,
        show_default=True,
        help="Number of conformations per tautomer",
    ),
    click.option(
        "--txt-out",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Write pH-specific microstate tables to this text file",
    ),
]


def cutoff_export_option(default: float = 0.2) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Add the cutoff-export option with a command-specific default."""

    return click.option(
        "--cutoff-export",
        type=float,
        default=default,
        show_default=True,
        help="Min. probability relative to the most probable microstate for export",
    )


def common_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Apply the Click options shared by all commands."""

    decorated_command = command

    for option in reversed(COMMON_OPTIONS):
        decorated_command = option(decorated_command)
    return decorated_command


def run_cli() -> None:
    """Main entry point of cli."""

    argv = sys.argv[1:]

    if not argv:
        cmd = "--help"
    else:
        cmd = argv[0]

    if (cmd not in ["--help", "-h"]) and (cmd not in COMMANDS):
        # Insert default command
        argv = ["single"] + argv
    cli(argv)


@click.group(
    cls=RejectDuplicateOptionsGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    pass


### Single molecule ###

@cli.command()
@click.option("--name", required=False, type=str, default="molecule", help="Molecule name")
@click.option("--smiles", required=True, type=str, help="SMILES string")
@click.option("--ph", "--pH", required=False, type=float, default=7.0, help="pH value")
@click.option("--sdf-out", required=False, type=click.Path(dir_okay=False, path_type=Path), help="sdf output file name")
@cutoff_export_option()
@common_options
def single(
    name: str,
    smiles: str,
    ph: float,
    sdf_out: Path,
    cutoff_export: float,
    model: str,
    unipka_model_folder: Path | None,
    nthreads: int,
    gpu: bool,
    matrix_def: str,
    cutoff_states: int,
    tautomer_search: bool,
    max_tautomers: int,
    num_confs: int,
    txt_out: Path | None,
) -> None:
    """Run single protonation state prediction given a smiles string and pH values."""

    _common_option_conflicts(click.get_current_context())
    _check_distinct_paths((
        ("--sdf-out", sdf_out),
        ("--txt-out", txt_out),
    ))

    smiles_out, mols_out = protonate(
        smiles,
        name=name,
        pH=ph,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        cutoff_export=cutoff_export,
        tautomer_search=tautomer_search,
        max_tautomers=max_tautomers,
        num_confs=num_confs,
        nthreads=nthreads,
        **_model_kwargs(model, unipka_model_folder, gpu),
    )

    click.echo(format_microstate_table(name, ph, smiles_out, mols_out))

    if txt_out:
        write_microstate_tables(txt_out, [(name, ph, smiles_out, mols_out)])

    if sdf_out:
        save_sdf(mols_out, sdf_out)


### Batch processing ###


@cli.command()
@click.option("--smi", required=True, type=click.Path(path_type=Path), help="Input .smi for batch processing")
@click.option("--ph", "--pH", required=False, type=float, default=7.0, help="pH value")
@click.option(
    "--overwrite/--no-overwrite",
    is_flag=True,
    default=True,
    help="Overwrite individual SDF files if they exist.",
)
@click.option(
    "--individual-sdfs/--no-individual-sdfs",
    default=False,
    show_default=True,
    help="Also write one SDF file per input molecule to --path-out.",
)
@click.option(
    "--path-out",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("batch_individual_sdfs"),
    show_default=True,
    help="Output folder used with --individual-sdfs.",
)
@click.option(
    "--sdf-combined",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("molecules_batch.sdf"),
    show_default=True,
    help=(
        "Write all exported microstates for all input molecules to one SDF file. "
        "A molecule can have multiple records when multiple microstates meet "
        "--cutoff-export; the default 1.0 keeps only the most probable state(s)."
    ),
)
@click.option(
    "--njobs",
    required=False,
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel jobs for batch processing. Set --nthreads (per job) accordingly to not overload.",
)
@cutoff_export_option(default=1.0)
@common_options
def batch(
    smi: Path,
    ph: float,
    path_out: Path,
    sdf_combined: Path,
    individual_sdfs: bool,
    overwrite: bool,
    cutoff_export: float,
    model: str,
    unipka_model_folder: Path | None,
    nthreads: int,
    gpu: bool,
    njobs: int,
    matrix_def: str,
    cutoff_states: int,
    tautomer_search: bool,
    max_tautomers: int,
    num_confs: int,
    txt_out: Path | None,
) -> None:
    """Batch process an input .smi file and write a combined SDF and stdout tables."""

    _common_option_conflicts(click.get_current_context())

    if not smi.is_file():
        raise click.BadParameter("must be an existing file", param_hint="--smi")

    try:
        batch_input = read_smi(smi)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--smi") from exc

    batch_paths: list[tuple[str, Path | None]] = [
        ("--smi", smi),
        ("--sdf-combined", sdf_combined),
        ("--txt-out", txt_out),
    ]
    if individual_sdfs:
        batch_paths.extend(
            (
                f"individual SDF for {name!r}",
                path_out / _safe_sdf_filename(idx, name),
            )
            for idx, name in enumerate(batch_input)
        )
    _check_distinct_paths(batch_paths)

    results_parallel = Parallel(n_jobs=njobs, prefer="processes")(
        delayed(_compute_protonate)(
            idx,
            smiles,
            name=name,
            pH=ph,
            matrix_def=matrix_def,
            cutoff_states=cutoff_states,
            cutoff_export=cutoff_export,
            tautomer_search=tautomer_search,
            max_tautomers=max_tautomers,
            num_confs=num_confs,
            nthreads=nthreads,
            **_model_kwargs(model, unipka_model_folder, gpu),
        )
        for idx, (name, smiles) in enumerate(batch_input.items()))

    batch_outputs = [
        (name, ph, smiles_out, mols_out)
        for _, name, smiles_out, mols_out in results_parallel
    ]
    formatted_output = format_microstate_tables(batch_outputs)
    if formatted_output:
        click.echo(formatted_output)

    combined_mols = tuple(
        mol
        for _, _, _, mols_out in results_parallel
        for mol in mols_out
    )
    save_sdf(combined_mols, sdf_combined)

    if individual_sdfs:
        os.makedirs(path_out, exist_ok=True)
        for idx, name, _, mols_out in results_parallel:
            filename = path_out / _safe_sdf_filename(idx, name)
            if (not overwrite) and (os.path.isfile(filename)):
                raise FileExistsError(f"File {filename} exists and overwrite == False!")
            save_sdf(mols_out, filename)

    if txt_out:
        write_microstate_tables(txt_out, batch_outputs)


### pH scan ###


@cli.command()
@click.option("--name", required=False, type=str, default="molecule", help="Molecule name")
@click.option("--smiles", required=True, type=str, help="SMILES string")
@click.option("--min-ph", required=False, type=float, default=0.0, help="Minimum pH value")
@click.option("--max-ph", required=False, type=float, default=14.0, help="Maximum pH value")
@click.option("--fig-out", required=False, type=click.Path(dir_okay=False, path_type=Path), help="Figure of scan")
@click.option("--sdf-out", required=False, type=click.Path(dir_okay=False, path_type=Path), help="File name for sdf output")
@click.option("--pkas-out", required=False, type=click.Path(dir_okay=False, path_type=Path), help="File for macro pkas")
@cutoff_export_option()
@common_options
def scan(
    name: str,
    smiles: str,
    min_ph: float,
    max_ph: float,
    fig_out: Path,
    sdf_out: Path,
    pkas_out: Path,
    model: str,
    unipka_model_folder: Path | None,
    nthreads: int,
    gpu: bool,
    matrix_def: str,
    cutoff_states: int,
    tautomer_search: bool,
    max_tautomers: int,
    num_confs: int,
    cutoff_export: float,
    txt_out: Path | None,
) -> None:
    """Scan pH values, output plot of microstate distributions and macro pKa values"""

    _common_option_conflicts(click.get_current_context())

    click.echo("Scan pH")

    import numpy as np

    pHs = np.arange(min_ph, max_ph + 0.0001, 0.25, dtype=np.float64)

    output_name = _safe_filename_stem(name)
    if not fig_out:
        fig_out = Path(f"{output_name}_scan.svg")
    if not sdf_out:
        sdf_out = Path(f"{output_name}_mols_scan.sdf")
    if not pkas_out:
        pkas_out = Path(f"{output_name}_macro_pkas.out")

    _check_distinct_paths((
        ("--fig-out", fig_out),
        ("--sdf-out", sdf_out),
        ("--pkas-out", pkas_out),
        ("--txt-out", txt_out),
    ))

    scan = scan_pH(
        smiles,
        name=name,
        pHs=pHs,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        cutoff_export=cutoff_export,
        tautomer_search=tautomer_search,
        max_tautomers=max_tautomers,
        num_confs=num_confs,
        nthreads=nthreads,
        **_model_kwargs(model, unipka_model_folder, gpu),
    )

    scan.export_macro_pkas(file=pkas_out)
    scan.print_macro_pkas()

    size_x = 150
    size_y = 120

    fig_scan = scan.plot_scan()
    fig_mols = scan.plot_mols(size_x=size_x, size_y=size_y)

    scan.export_scan(fig_out, fig_scan, fig_mols)
    scan.save_sdf(sdf_out)

    if txt_out:
        scan_outputs = []
        for pH in scan.pHs:
            molecule = scan.molecule_at(float(pH))
            scan_outputs.append((name, float(pH), molecule.smiles, molecule.mols))
        write_microstate_tables(txt_out, scan_outputs)
