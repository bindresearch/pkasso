import os

file = '/home/sobuelow/projects/asdf_ligands/Cmp_trial_asdf/asyn_asdf_mod.smi'

print(f'Using {file}')

with open(file,'r') as f:
    for line in f:
        print('================')
        spl = line.split()
        name = spl[1]
        print(name)
        os.system(f'python run_autoprot.py --file {file} --name {name}')
