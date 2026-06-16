"""Command-line interface for running pKasso workflows."""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import numpy as np
from numpy.typing import NDArray

from .py_interface import protonate, scan_pH
from .utils import read_smi
from .postprocess import save_sdf

COMMANDS = {"single", "batch", "scan"}

def _common_option_conflicts(ctx: click.Context) -> None:
    """Raise a Click error for explicitly incompatible shared CLI options."""

    commandline = click.core.ParameterSource.COMMANDLINE
    max_tautomers_source = ctx.get_parameter_source("max_tautomers")
    tautomer_search_source = ctx.get_parameter_source("tautomer_search")
    num_confs_source = ctx.get_parameter_source("num_confs")

    if (
        max_tautomers_source == commandline
        and tautomer_search_source == commandline
        and ctx.params["tautomer_search"] is False
    ):
        raise click.UsageError("--max-tautomers cannot be used with --no-tautomer-search.")

    if (
        num_confs_source == commandline
        and tautomer_search_source == commandline
        and ctx.params["tautomer_search"] is False
    ):
        raise click.UsageError("--num-confs cannot be used with --no-tautomer-search.")

    if ctx.params["cutoff_states"] < 1:
        raise click.UsageError("--cutoff-states must be >= 1.")

    if (ctx.params["cutoff_export"] < 0) or (ctx.params["cutoff_export"] > 1):
        raise click.UsageError("--cutoff-export must be >= 0 and <= 1.")



COMMON_OPTIONS = [
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
]


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


@click.group()
def cli() -> None:
    pass


### Single molecule ###

@cli.command()
@click.option("--name", required=False, type=str, default="molecule", help="Molecule name")
@click.option("--smiles", required=True, type=str, help="SMILES string")
@click.option("--ph", required=False, type=float, default=7.0, help="pH value (for sdf and csv output)")
@click.option("--sdf-out", required=False, type=click.Path(path_type=Path), help="sdf output file name")
@click.option(
    "--cutoff-export",
    required=False,
    type=float,
    default=1.0,
    show_default=True,
    help="Min. probability of microstate w.r.t. highest probable microstate to be included for export",
)
@common_options
def single(
    name: str,
    smiles: str,
    ph: float,
    sdf_out: Path,
    cutoff_export: float,
    matrix_def: str,
    cutoff_states: int,
    tautomer_search: bool,
    max_tautomers: int,
    num_confs: int,
) -> None:
    """Run single protonation state prediction given a smiles string and pH values."""

    _common_option_conflicts(click.get_current_context())

    # click.echo(f"Single: {name}")

    smiles_out, mols_out = protonate(
        smiles,
        pH=ph,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        cutoff_export=cutoff_export,
        tautomer_search=tautomer_search,
        max_tautomers=max_tautomers,
        num_confs=num_confs,
    )
    print(f"{name} | pH: {ph}")
    print("Microstate SMILES Probability Net_Charge")
    print("----------------------------------------")
    for sm, mol in zip(smiles_out, mols_out):
        name_state = mol.GetProp("_Name")
        probability = float(mol.GetProp("Probability"))
        net_charge = float(mol.GetProp("net_charge"))
        print(name_state, sm, f"{probability:.5f}", net_charge)

    if sdf_out:
        save_sdf(mols_out, sdf_out)


### Batch processing ###


@cli.command()
@click.option("--smi", required=True, type=click.Path(path_type=Path), help="Input .smi for batch processing")
@click.option("--ph", required=False, type=float, default=7.0, help="pH value (for sdf and csv output)")
@click.option("--overwrite/--no-overwrite", is_flag=True, default=True, help="Overwrite sdf file if exists")
@click.option(
    "--path-out",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("pkasso_output"),
    help="Output folder for sdf files",
)
@click.option(
    "--cutoff-export",
    required=False,
    type=float,
    default=1.0,
    show_default=True,
    help="Min. probability of microstate w.r.t. highest probable microstate to be included for export",
)
@common_options
def batch(
    smi: Path,
    ph: float,
    path_out: Path,
    overwrite: bool,
    cutoff_export: float,
    matrix_def: str,
    cutoff_states: int,
    tautomer_search: bool,
    max_tautomers: int,
    num_confs: int,
) -> None:
    """Batch process an input .smi file and write output microstates to csv
    (optionally write sdf files of individual molecules)"""

    _common_option_conflicts(click.get_current_context())

    batch_input = read_smi(smi)

    for name, smiles in batch_input.items():
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
        )
        print(name, smiles_out)

        # Save sdf files
        if path_out:
            os.makedirs(path_out, exist_ok=True)
            filename = path_out / f"{name}.sdf"
            if (not overwrite) and (os.path.isfile(filename)):
                raise FileExistsError("File {file_name} exists and overwrite == False!")
            save_sdf(mols_out, filename)


### pH scan ###


@cli.command()
@click.option("--name", required=False, type=str, default="molecule", help="Molecule name")
@click.option("--smiles", required=True, type=str, help="SMILES string")
@click.option("--min-ph", required=False, type=float, default=0.0, help="Minimum pH value")
@click.option("--max-ph", required=False, type=float, default=14.0, help="Maximum pH value")
@click.option("--fig-out", required=False, type=click.Path(path_type=Path), help="Figure of scan")
@click.option("--sdf-out", required=False, type=click.Path(path_type=Path), help="File name for sdf output")
@click.option("--pkas-out", required=False, type=click.Path(path_type=Path), help="File for macro pkas")
@common_options
def scan(
    name: str,
    smiles: str,
    min_ph: float,
    max_ph: float,
    fig_out: Path,
    sdf_out: Path,
    pkas_out: Path,
    matrix_def: str,
    cutoff_states: int,
    tautomer_search: bool,
    max_tautomers: int,
    num_confs: int,
) -> None:
    """Scan pH values, output plot of microstate distributions and macro pKa values"""

    _common_option_conflicts(click.get_current_context())

    click.echo("Scan pH")

    pHs: NDArray[np.float64] = np.arange(min_ph, max_ph + 0.0001, 0.25, dtype=np.float64)

    if not fig_out:
        fig_out = Path(f"{name}_scan.svg")
    if not sdf_out:
        sdf_out = Path(f"{name}_mols_scan.sdf")
    if not pkas_out:
        pkas_out = Path(f"{name}_macro_pkas.out")

    scan = scan_pH(
        smiles,
        pHs=pHs,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        tautomer_search=tautomer_search,
        max_tautomers=max_tautomers,
        num_confs=num_confs,
    )

    scan.export_macro_pkas(file=pkas_out)
    scan.print_macro_pkas()

    size_x = 150
    size_y = 120

    fig_scan = scan.plot_scan()
    fig_mols = scan.plot_mols(size_x=size_x, size_y=size_y)

    scan.export_scan(fig_out, fig_scan, fig_mols)
    scan.save_sdf(sdf_out)
