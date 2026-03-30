from .py_interface import protonate, batch_protonate, scan_pH

import sys
import os
import click

import numpy as np
from numpy.typing import NDArray

from typing import Any

COMMANDS = {"single", "batch", "scan"}

COMMON_OPTIONS = [
    click.option("--matrix-def", type=click.Choice(["dG", "msm"]), default="dG", show_default=True),
    click.option("--cutoff-states", type=int, default=4000, show_default=True),
    click.option("--sfreq-cutoff-individual", type=float, default=0.01, show_default=True),
    click.option("--sfreq-cutoff-combined", type=float, default=0.001, show_default=True),
    click.option("--ph-band", type=float, default=10.0, show_default=True),
    click.option("--cutoff-export", type=float, default=0.2, show_default=True),
]

def common_options(f: Any) -> Any:
    """ Collect options shared by single, batch, scan. """

    for opt in reversed(COMMON_OPTIONS):
        f = opt(f)
    return f

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
@click.option('--name', required=True, type=str, help='Molecule name: str')
@click.option('--smiles', required=True, type=str, help='SMILES string: str')
@click.option('--ph', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')
@click.option('--out', required=False, type=str, help='sdf output file name: str')
@common_options
# @click.pass_context
def single(name: str, smiles: str, ph: float, out: str, matrix_def: str, cutoff_states: int, sfreq_cutoff_individual: float,
           sfreq_cutoff_combined: float, ph_band: float, cutoff_export: float) -> None:
    """ Run single protonation state prediction given a smiles string and pH values. """

    click.echo(f"Single: {name}")
    if not out:
        out = f'{name}_output.sdf'
    molecule = protonate(smiles, pH=ph, matrix_def=matrix_def, cutoff_states=cutoff_states, sfreq_cutoff_individual=sfreq_cutoff_individual,
                         sfreq_cutoff_combined=sfreq_cutoff_combined, pH_band=ph_band, cutoff_export=cutoff_export)
    print(f'{name} | pH: {ph}')
    print('Microstate SMILES Probability')
    for m in molecule.microstates:
        print(m.name_state, m.smiles, m.freq)
    molecule.save(out)

### Batch processing ###

@cli.command()
@click.option('--smi', required=True, type=str, default='input.csv', help='Input .smi for batch processing: str')
@click.option('--ph', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')
@click.option('--csv-out', required=False, type=str, default='batch_results.csv', help='Output file for summary csv: str')
@click.option('--save-sdf', is_flag=True, default=True, help='Save sdf file for each molecule: bool')
@click.option('--sdf-folder', required=False, type=str, default='output', help='Output folder for sdf files: str')
@common_options
# @click.pass_context
def batch(smi: str, ph: float, csv_out: str, save_sdf: bool, sdf_folder: str, 
          matrix_def: str, cutoff_states: int, sfreq_cutoff_individual: float,
           sfreq_cutoff_combined: float, ph_band: float, cutoff_export: float) -> None:
    """ Batch process an input .smi file and write output microstates to csv 
    (optionally write sdf files of individual molecules)"""

    click.echo("Batch")
    bat = batch_protonate(smi, pH=ph, matrix_def=matrix_def, cutoff_states=cutoff_states, sfreq_cutoff_individual=sfreq_cutoff_individual,
                         sfreq_cutoff_combined=sfreq_cutoff_combined, pH_band=ph_band, cutoff_export=cutoff_export)
    df = bat.to_pandas()
    df.to_csv(csv_out)
    if save_sdf:
        os.makedirs(sdf_folder, exist_ok=True)
        for name, molecule in bat.molecules.items():
            molecule.save(f'{sdf_folder}/{name}.sdf')

### pH scan ###

@cli.command()
@click.option('--name',required=True, type=str, help='Molecule name: str')
@click.option('--smiles', required=True, type=str, help='SMILES string: str')
@click.option('--min-ph', required=False, type=float, default=0., help='Minimum pH value')
@click.option('--max-ph', required=False, type=float, default=14., help='Maximum pH value')
@click.option('--fig_out', required=False, type=str, help='Figure of scan: str')
@click.option('--sdf_out', required=False, type=str, help='File name for sdf output: str')
@click.option('--pkas_out', required=False, type=str, help='File for macro pkas: str')
@common_options
def scan(name:str , smiles: str, min_ph: float, max_ph: float, 
         matrix_def: str, cutoff_states: int, sfreq_cutoff_individual: float,
           sfreq_cutoff_combined: float, ph_band: float, cutoff_export: float, 
           fig_out: str, sdf_out: str, pkas_out: str) -> None:
    """ Scan pH values, output plot of microstate distributions and macro pKa values """

    click.echo("Scan pH")

    pHs: NDArray[np.float64] = np.arange(min_ph, max_ph+0.0001, 0.25, dtype=np.float64)

    if not fig_out:
        fig_out = f'{name}_scan.pdf'
    if not sdf_out:
        sdf_out = f'{name}_mols_scan.sdf'
    if not pkas_out:
        pkas_out = f'{name}_macro_pkas.out'

    scan = scan_pH(
        smiles,
        pHs = pHs,
        matrix_def=matrix_def, cutoff_states=cutoff_states, sfreq_cutoff_individual=sfreq_cutoff_individual,
                         sfreq_cutoff_combined=sfreq_cutoff_combined, pH_band=ph_band, cutoff_export=cutoff_export)
    
    scan.export_macro_pkas(file=pkas_out)
    scan.print_macro_pkas()

    size_x = 150
    size_y = 120

    fig_scan = scan.plot_scan()
    fig_mols = scan.plot_mols(size_x = size_x, size_y=size_y)

    scan.export_scan(fig_out, fig_scan, fig_mols)
    scan.save_sdf(sdf_out)