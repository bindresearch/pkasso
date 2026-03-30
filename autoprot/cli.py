# # from autoprot.main import Autoprot

# from py_interface import protonate, batch_protonate, scan_pH

# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# import numpy as np
# from numpy.typing import NDArray

# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# parser.add_argument('--mode', required=False, type=str, choices=['single','batch','pH_scan'], default='single', help='Run mode: str')
# parser.add_argument('--pH', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')
# parser.add_argument('-s', '--smiles',required=True, type=str, help='REQUIRED: SMILES string: str')
# parser.add_argument('-n', '--name',required=True, type=str, help='REQUIRED: Molecule name: str')

# Single options
# parser.add_argument('--out', required=False, type=str, default='mol_output.sdf', help='sdf output file name: str')

# Batch options
# parser.add_argument('--pH', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')
# parser.add_argument('--smi', required=False, type=str, default='input.csv', help='Input .smi for batch processing: str')
# parser.add_argument('--save_sdf', required=False, action="store_true", help='Save sdf file for each molecule: bool')
# parser.add_argument('--sdf_folder', required=False, type=str, default='output', help='Output folder for sdf files: str')
# parser.add_argument('--csv', required=False, type=str, default='autoprot_results.csv', help='Output file for summary csv: str')

# pH_scan options
# parser.add_argument('--pHs', required=False, type=NDArray[np.float64], default=np.arange(0, 14.1, 0.5), help='pH values to scan: NDArray[np.float64]')

# Internal options
# parser.add_argument('--matrix_def', required=False, type=str, choices=['dG','msm'], default='dG', help='Use free energy differences of MSM to compare microstates: str')
# parser.add_argument('--cutoff_states', required=False, type=int, default=4000, help='Max number of coupled protonable sites: int')
# parser.add_argument('--sfreq_cutoff_individual', required=False, type=float, default=0.01, help='Frequency of individual cluster to use for combined clusters: float')
# parser.add_argument('--sfreq_cutoff_combined', required=False, type=float, default=0.001, help='Frequency of microstate to include: float')
# parser.add_argument('--pH_band', required=False, type=float, default=10., help='Band of pKa values to include in calculation around a given pH: float')
# parser.add_argument('--cutoff_export', required=False, type=float, default=0.2, help='Microstate frequency cutoff (relative to max frequency microstate) to include in csv file output: float')

# help='Input .smi for batch processing (ignores --smiles): str')

# parser.add_argument('--batch', required=False, type=str, default='', help='Input .smi for batch processing (ignores --smiles): str')

# parser.add_argument('--outfolder', required=False, type=str, default='output', help='Output path for output data and figures: str')
# parser.add_argument('--fout_csv', required=False, type=str, default='autoprot_results.csv', help='Filename for output csv file: str')
# parser.add_argument('--append_csv', required=False, action="store_true", help='Append to csv (do not overwrite): bool')
# parser.add_argument('--no_sdf', required=False, action="store_true", help='Do not write sdf structure files of highest frequency microstates: bool')

# parser.add_argument('--pH_scan', required=False, action="store_true", help='Run pH scan: bool')

# parser.add_argument('-v', '--verbose',required=False, action="store_true", help='verbose: bool')

# ARGS = parser.parse_args()

# def run_cli():
#     args = ARGS
#     if args.mode == 'single':
#         molecule = protonate(args.smiles, pH = args.pH)
#         molecule.save(args.out)
#     elif args.mode == 'batch':
#         batch = batch_protonate(args.)
#         os.makedirs(self.path, exist_ok=True)

#     ap = Autoprot(
#         name=args.name, smiles=args.smiles,
#         mode=args.mode,
#         outfolder = args.outfolder, fout_csv=args.fout_csv, 
#         append_csv=args.append_csv, no_sdf=args.no_sdf,
#         cutoff_export = args.cutoff_export,
#         pH = args.pH,
        
#         pH_band=args.pH_band, pHs=args.pHs,
#         matrix_def=args.matrix_def, 
#         cutoff_states=args.cutoff_states, sfreq_cutoff_individual=args.sfreq_cutoff_individual, 
#         sfreq_cutoff_combined=args.sfreq_cutoff_combined,
#         verbose=args.verbose)
#     print('='*20)
#     print(f'Input SMILES: {ap.smiles}')
#     ap.run()
#     print('-'*20)
#     print(f'pH: {ap.pH}')
#     for smiles, sfreq in zip(ap.smiles_out, ap.sfreqs_out):
#         print(f'{smiles} | Freq: {sfreq:.3f}')

# if __name__ == '__main__':
#     run_cli()

from .py_interface import protonate, batch_protonate, scan_pH

import sys
import os
import click

import numpy as np
from numpy.typing import NDArray

COMMANDS = {"single", "batch", "scan"}

COMMON_OPTIONS = [
    click.option("--matrix-def", type=click.Choice(["dG", "msm"]), default="dG", show_default=True),
    click.option("--cutoff-states", type=int, default=4000, show_default=True),
    click.option("--sfreq-cutoff-individual", type=float, default=0.01, show_default=True),
    click.option("--sfreq-cutoff-combined", type=float, default=0.001, show_default=True),
    click.option("--ph-band", type=float, default=10.0, show_default=True),
    click.option("--cutoff-export", type=float, default=0.2, show_default=True),
]

def common_options(f):
    for opt in reversed(COMMON_OPTIONS):
        f = opt(f)
    return f

def run_cli():
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
def cli():
    pass

@cli.command()
@click.option('--name', required=True, type=str, help='Molecule name: str')
@click.option('--smiles', required=True, type=str, help='SMILES string: str')
@click.option('--ph', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')
@click.option('--out', required=False, type=str, help='sdf output file name: str')
@common_options
# @click.pass_context
def single(name, smiles, ph, out, matrix_def, cutoff_states, sfreq_cutoff_individual,
           sfreq_cutoff_combined, ph_band, cutoff_export):
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

@cli.command()
@click.option('--smi', required=True, type=str, default='input.csv', help='Input .smi for batch processing: str')
@click.option('--ph', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')
@click.option('--csv-out', required=False, type=str, default='batch_results.csv', help='Output file for summary csv: str')
@click.option('--save-sdf', is_flag=True, default=True, help='Save sdf file for each molecule: bool')
@click.option('--sdf-folder', required=False, type=str, default='output', help='Output folder for sdf files: str')
@common_options
# @click.pass_context
def batch(smi, ph, csv_out, save_sdf, sdf_folder, matrix_def, cutoff_states, sfreq_cutoff_individual,
           sfreq_cutoff_combined, ph_band, cutoff_export):
    click.echo("Batch")
    bat = batch_protonate(smi, pH=ph, matrix_def=matrix_def, cutoff_states=cutoff_states, sfreq_cutoff_individual=sfreq_cutoff_individual,
                         sfreq_cutoff_combined=sfreq_cutoff_combined, pH_band=ph_band, cutoff_export=cutoff_export)
    df = bat.to_pandas()
    df.to_csv(csv_out)
    if save_sdf:
        os.makedirs(sdf_folder, exist_ok=True)
        for name, molecule in bat.molecules.items():
            molecule.save(f'{sdf_folder}/{name}.sdf')

@cli.command()
@click.option('--name',required=True, type=str, help='Molecule name: str')
@click.option('--smiles', required=True, type=str, help='SMILES string: str')
@click.option('--min-ph', required=False, type=float, default=0., help='Minimum pH value')
@click.option('--max-ph', required=False, type=float, default=14., help='Maximum pH value')
@click.option('--fig_out', required=False, type=str, help='Figure of scan: str')
@click.option('--sdf_out', required=False, type=str, help='File name for sdf output: str')
@click.option('--pkas_out', required=False, type=str, help='File for macro pkas: str')
@common_options
def scan(name, smiles, min_ph, max_ph, matrix_def, cutoff_states, sfreq_cutoff_individual,
           sfreq_cutoff_combined, ph_band, cutoff_export, fig_out, sdf_out, pkas_out):
    click.echo("Scan pH")
    pHs: NDArray[np.float64] = np.arange(min_ph, max_ph+0.0001, 0.25)

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