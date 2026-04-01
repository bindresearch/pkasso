"""Command-line interface for running AutoProt workflows."""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import numpy as np
from numpy.typing import NDArray

from .py_interface import batch_protonate, protonate, scan_pH

COMMANDS = {"single", "batch", "scan"}

COMMON_OPTIONS = [
    click.option(
        "--matrix-def",
        type=click.Choice(["dG", "msm"]),
        default="dG",
        show_default=True,
        help='Use free energy differences or Markov state model to determine microstate probabilities',
    ),
    click.option(
        "--cutoff-states",
        type=int,
        default=4000,
        show_default=True,
        help='Max. number of microstates per coupled cluster of protonation sites',
    ),
    click.option(
        "--sfreq-cutoff-individual",
        type=float,
        default=0.01,
        show_default=True,
        help='Min. probability of protonation site cluster to be included in final microstate combination',
    ),
    click.option(
        "--sfreq-cutoff-combined",
        type=float,
        default=0.001,
        show_default=True,
        help='Min. probability of combined microstate (from independent clusters) to be considered',
    ),
    click.option(
        "--ph-band",
        type=float,
        default=10.0,
        show_default=True,
        help='Allowed pKa tolerance around the pH when determining candidate sites',
    ),
]

def common_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Apply the Click options shared by all commands."""

    decorated_command = command

    for option in reversed(COMMON_OPTIONS):
        decorated_command = option(decorated_command)
    return decorated_command

def run_cli() -> None:
    """ Main entry point of cli. """

    argv = sys.argv[1:]

    if not argv:
        cmd = '--help'
    else:
        cmd = argv[0]

    if (cmd not in ['--help', '-h']) and (cmd not in COMMANDS):
        # Insert default command
        argv = ["single"] + argv
    cli(argv)

@click.group()
def cli() -> None:
    pass

### Single molecule ###

@cli.command()
@click.option('--name', required=False, type=str, default='molecule', help='Molecule name')
@click.option('--smiles', required=True, type=str, help='SMILES string')
@click.option('--ph', required=False, type=float, default=7., help='pH value (for sdf and csv output)')
@click.option('--out', required=False, type=click.Path(path_type=Path), help='sdf output file name')
@click.option(
    "--cutoff-export",
    type=float,
    default=0.2,
    show_default=True,
    help='Min. probability of microstate w.r.t. highest probable microstate to be included for export',
)
@common_options
# @click.pass_context
def single(
    name: str,
    smiles: str,
    ph: float,
    out: Path,
    cutoff_export: float,
    matrix_def: str,
    cutoff_states: int,
    sfreq_cutoff_individual: float,
    sfreq_cutoff_combined: float,
    ph_band: float,
) -> None:
    """ Run single protonation state prediction given a smiles string and pH values. """

    click.echo(f"Single: {name}")
    if not out:
        out_path = Path(f'{name}_output.sdf')
    else:
        out_path = Path(out)

    molecule = protonate(
        smiles,
        pH=ph,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        sfreq_cutoff_individual=sfreq_cutoff_individual,
        sfreq_cutoff_combined=sfreq_cutoff_combined,
        pH_band=ph_band,
        cutoff_export=cutoff_export
    )
    print(f'{name} | pH: {ph}')
    print('Microstate SMILES Probability')
    for m in molecule.microstates:
        print(m.name_state, m.smiles, m.freq)
    molecule.save(out_path)

### Batch processing ###

@cli.command()
@click.option(
    '--smi',
    required=True,
    type=click.Path(path_type=Path),
    help='Input .smi for batch processing'
)
@click.option(
    '--ph',
    required=False,
    type=float,
    default=7.,
    help='pH value (for sdf and csv output)'
)
@click.option(
    '--csv-out',
    required=False,
    type=click.Path(path_type=Path),
    default=Path('batch_results.csv'),
    help='Output file for summary csv'
)
@click.option(
    '--save-sdf',
    is_flag=True,
    default=True,
    help='Save sdf file for each molecule'
)
@click.option(
    '--sdf-folder',
    required=False,
    type=click.Path(path_type=Path),
    default=Path('output'),
    help='Output folder for sdf files'
)
@click.option(
    "--cutoff-export",
    type=float,
    default=0.2,
    show_default=True,
    help='Min. probability of microstate w.r.t. highest probable microstate to be included for export',
)
@common_options
# @click.pass_context
def batch(
    smi: Path,
    ph: float,
    csv_out: Path,
    save_sdf: bool,
    sdf_folder: Path,
    cutoff_export: float,
    matrix_def: str,
    cutoff_states: int,
    sfreq_cutoff_individual: float,
    sfreq_cutoff_combined: float,
    ph_band: float,
    
) -> None:
    """ Batch process an input .smi file and write output microstates to csv 
    (optionally write sdf files of individual molecules)"""

    click.echo("Batch")
    bat = batch_protonate(
        smi,
        pH=ph,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        sfreq_cutoff_individual=sfreq_cutoff_individual,
        sfreq_cutoff_combined=sfreq_cutoff_combined,
        pH_band=ph_band,
        cutoff_export=cutoff_export
    )
    df = bat.to_pandas()
    df.to_csv(csv_out)
    if save_sdf:
        os.makedirs(sdf_folder, exist_ok=True)
        for name, molecule in bat.molecules.items():
            molecule.save(sdf_folder / f'{name}.sdf')

### pH scan ###

@cli.command()
@click.option('--name', required=False, type=str, default='molecule', help='Molecule name')
@click.option('--smiles', required=True, type=str, help='SMILES string')
@click.option('--min-ph', required=False, type=float, default=0., help='Minimum pH value')
@click.option('--max-ph', required=False, type=float, default=14., help='Maximum pH value')
@click.option('--fig-out', required=False, type=click.Path(path_type=Path), help='Figure of scan')
@click.option('--sdf-out', required=False, type=click.Path(path_type=Path), help='File name for sdf output')
@click.option('--pkas-out', required=False, type=click.Path(path_type=Path), help='File for macro pkas')
@common_options
def scan(
    name:str,
    smiles: str,
    min_ph: float,
    max_ph: float,
    matrix_def: str,
    cutoff_states: int,
    sfreq_cutoff_individual: float,
    sfreq_cutoff_combined: float,
    ph_band: float,
    cutoff_export: float, 
    fig_out: Path,
    sdf_out: Path,
    pkas_out: Path,
) -> None:
    """ Scan pH values, output plot of microstate distributions and macro pKa values """

    click.echo("Scan pH")

    pHs: NDArray[np.float64] = np.arange(min_ph, max_ph+0.0001, 0.25, dtype=np.float64)

    if not fig_out:
        fig_out = Path(f'{name}_scan.pdf')
    if not sdf_out:
        sdf_out = Path(f'{name}_mols_scan.sdf')
    if not pkas_out:
        pkas_out = Path(f'{name}_macro_pkas.out')

    scan = scan_pH(
        smiles,
        pHs = pHs,
        matrix_def=matrix_def,
        cutoff_states=cutoff_states,
        sfreq_cutoff_individual=sfreq_cutoff_individual,
        sfreq_cutoff_combined=sfreq_cutoff_combined,
        pH_band=ph_band,
        cutoff_export=cutoff_export
    )
    
    scan.export_macro_pkas(file=pkas_out)
    scan.print_macro_pkas()

    size_x = 150
    size_y = 120

    fig_scan = scan.plot_scan()
    fig_mols = scan.plot_mols(size_x = size_x, size_y=size_y)

    scan.export_scan(fig_out, fig_scan, fig_mols)
    scan.save_sdf(sdf_out)