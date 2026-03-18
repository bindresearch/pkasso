from autoprot.main import Autoprot
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from numpy.typing import NDArray

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-n', '--name',required=True, type=str, help='REQUIRED: Molecule name: str')
parser.add_argument('-s', '--smiles',required=True, type=str, help='REQUIRED: SMILES string: str')
parser.add_argument('--pH_output', required=False, type=float, default=7., help='pH value (for sdf and csv output): float')

parser.add_argument('--path_out', required=False, type=str, default='output', help='Output path for sdf and csv files: str')
parser.add_argument('--path_figs', required=False, type=str, default='figures', help='Output path for figures: str')
parser.add_argument('--fout_csv', required=False, type=str, default='autoprot_results.csv', help='Filename for output csv file: str')
parser.add_argument('--append_csv', required=False, action="store_true", help='Append to csv (do not overwrite): bool')
parser.add_argument('--no_sdf', required=False, action="store_true", help='Do not write sdf structure files of highest frequency microstates: bool')
parser.add_argument('--cutoff_export', required=False, type=float, default=0.2, help='Microstate frequency cutoff (relative to max frequency microstate) to include in csv file output: float')

parser.add_argument('--pH_band', required=False, type=float, default=10., help='Band of pKa values to include in calculation around a given pH: float')
parser.add_argument('--pHs', required=False, type=NDArray[np.float64], default=np.arange(0, 14.1, 0.5), help='pH values to scan: NDArray[np.float64]')

parser.add_argument('--matrix_def', required=False, type=str, choices=['dG','msm'], default='dG', help='Use free energy differences of MSM to compare microstates: str')

parser.add_argument('--cutoff_states', required=False, type=int, default=4000, help='Max number of coupled protonable sites: int')
parser.add_argument('--sfreq_cutoff_individual', required=False, type=float, default=0.01, help='Frequency of individual cluster to use for combined clusters: float')
parser.add_argument('--sfreq_cutoff_combined', required=False, type=float, default=0.001, help='Frequency of microstate to include: float')

parser.add_argument('-v', '--verbose',required=False, action="store_true", help='verbose: bool')

args = parser.parse_args()

if __name__ == '__main__':
    ap = Autoprot(
        name=args.name, smiles=args.smiles, 
        path_out = args.path_out, path_figs = args.path_figs, fout_csv=args.fout_csv, 
        append_csv=args.append_csv, no_sdf=args.no_sdf,
        cutoff_export = args.cutoff_export,
        pH_output = args.pH_output, pH_band=args.pH_band, pHs=args.pHs,
        matrix_def=args.matrix_def, 
        cutoff_states=args.cutoff_states, sfreq_cutoff_individual=args.sfreq_cutoff_individual, 
        sfreq_cutoff_combined=args.sfreq_cutoff_combined,
        verbose=args.verbose)
    print('='*20)
    print(f'Input SMILES: {ap.smiles}')
    ap.run()
    print('-'*20)
    print(f'pH: {ap.pH_output}')
    for smiles, sfreq in zip(ap.smiles_out, ap.sfreqs_out):
        print(f'{smiles} | Freq: {sfreq:.3f}')