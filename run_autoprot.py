from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from autoprot import AutoProt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',required=True,type=str)
parser.add_argument('--file',required=True,type=str)
args = parser.parse_args()

path_inp_qupkake = 'qupkake_input'
path_out_qupkake = 'qupkake_output'

path_out_autoprot = 'notauto_autoprot_output_fida'

name = args.name
f_input = args.file
ncycles = 20
cutoff = 0.1 # cutoff to try protonated state

# f_input = '/home/sobuelow/projects/martini_ligands/Cmp_trial_Fida/asyn_fida_mod.smi' # '../asyn_ligands_robustelli.smi' # 'test.smi' # 'test.smi' # 'naadp.smi' # 
# f_input = '/home/sobuelow/projects/martini_ligands/asyn_ligands_robustelli.smi'

pH = 7.5 # 7.4

verbose = False

autoprot = AutoProt(name,f_input=f_input,path_inp_qupkake=path_inp_qupkake,path_out_qupkake=path_out_qupkake,
                    path_out_autoprot=path_out_autoprot,pH=pH,
                    cutoff=cutoff,ncycles=ncycles,verbose=verbose)

autoprot.run_predictions()
autoprot.run_state_analysis()
autoprot.export_results()
