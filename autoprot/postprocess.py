import numpy as np
import matplotlib.pyplot as plt

from svgutils.compose import Figure, SVG
import cairosvg # import before anything in rdkit.Chem.Draw, breaks otherwise !!

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToFile, MolsToGridImage

import copy
import os

def plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, sfreqs_not_relevant, pkas_combined, cmap=plt.cm.Spectral,
                 path='figures',verbose=False):
    
    cmap = plt.cm.Spectral_r
    if len(pHs) == 1:
        style = 'o'
    else:
        style = '-'

    fsave = f'{path}/{name}_ph_scan.svg'

    if verbose:
        print(f'Indices: {indices}')
    px = 1/plt.rcParams['figure.dpi']

    fig, ax = plt.subplots(2,1,figsize=(700*px,500*px),height_ratios=[0.6,0.4])

    for idx, sfreq in enumerate(sfreqs_not_relevant):
        ax[0].plot(pHs,sfreq*100,style,color='gray',lw=1.,alpha=0.3)

    for idx, (state_str, sfreq) in enumerate(zip(state_strs_relevant,sfreqs_relevant)):
        if len(state_strs_relevant) > 1:
            color = cmap(idx/(len(state_strs_relevant)-1))
        else:
            color = cmap(0)
        ax[0].plot(pHs,sfreq*100,style,label=state_str,color=color)
    if len(state_strs_relevant) > 10:
        ax[0].legend(ncol=2,fontsize=6)
    elif len(state_strs_relevant) > 1:
        ax[0].legend(ncol=1,fontsize=8)

    ax[0].set(xlabel='pH',ylabel='Distribution [%]')
    
    ax[0].grid(alpha=0.3)

    ax[1].plot(pHs,net_charges,style,color='black')

    for idx, (q, pka) in enumerate(pkas_combined.items()):
        x = np.argmin(np.abs(pHs-pka))
        if q+1 > 0:
            color = 'tab:blue'
        else:
            color = 'tab:red'
        ax[1].plot(pHs[x],net_charges[x],'o',color=color,markersize=5)
        ax[1].text(pHs[x]+0.1,net_charges[x]+0.05,f'{pka:.2f}')


    ax[1].set(xlabel='pH', ylabel='Net charge')
    ax[1].grid(alpha=0.3)

    if len(pHs) > 1:
        for idx in range(2):
            ax[idx].set(xlim=(pHs[0],pHs[-1]))
            ax[idx].set_xticks(np.arange(pHs[0],pHs[-1]+0.001,1))

    fig.tight_layout()
    if fsave != '':
        fig.savefig(fsave, transparent=True)
    plt.close()
    # return fig

def export_sdf(state_strs,mols_lib,p):
    with Chem.SDWriter(f'{p.path}/{p.name}.sdf') as f:
        for e_idx, state_str in enumerate(state_strs):
            mol = mols_lib[state_str]
            mol_h = Chem.AddHs(mol)

            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)
            if cid != 0:
                raise ValueError(f'{p.name}_{state_str} could not be embedded.')
            AllChem.UFFOptimizeMolecule(mol_h)
            mol_h.SetProp("_Name", f'{p.name}_{e_idx}')
            f.write(mol_h)

def export_csv(state_strs,smiles_lib,sfreqs,state_qs,p):
    if p.append:
        action = 'a'
    else:
        action = 'w'
    with open(f'{p.path_out}/{p.fout_csv}',action) as f:
        if action == 'w':
            f.write(f'name,name_state,SMILES,frequency,charge\n')
        for e_idx, (state_str, sfreq) in enumerate(zip(state_strs, sfreqs)):
            smiles = smiles_lib[state_str]
            # Remove labels
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            smiles = Chem.MolToSmiles(mol)
            f.write(f'{p.name},{p.name}_{e_idx},{smiles},{sfreq/np.sum(sfreqs):.5f},{state_qs[state_str]}\n')

def export_macro_pkas(pkas_combined,p):
    with open(f'{p.path_out}/{p.name}_pkas.csv','w') as f:
        f.write('idx,q0,q1,pka\n')
        for idx, (q, pka) in enumerate(pkas_combined.items()):
            f.write(f'pKa{idx+1},{q},{q+1},{pka:.5f}\n')

def calc_relevant_states(state_freqs_all, mols_lib, max_states=18,verbose=False):
    """ Reduce number of states to max_states for plotting """

    if len(state_freqs_all.keys()) == 0:
        return 0, [], [], []

    cutoff = 0.01
    tries = 0

    N_relevant_states = 1e5
    while N_relevant_states > max_states:
        state_strs_relevant = []
        sfreqs_relevant = []
        sfreqs_not_relevant = []
        mols_relevant = []
        pH_argmaxs = []

        for state_str, sfreqs in state_freqs_all.items():
            if np.max(sfreqs) > cutoff:
                state_strs_relevant.append(state_str)
                sfreqs_relevant.append(sfreqs)
                mols_relevant.append(mols_lib[state_str])
                pH_argmaxs.append(np.argmax(sfreqs))
            else:
                sfreqs_not_relevant.append(sfreqs)
        N_relevant_states = len(state_strs_relevant)
        tries += 1
        cutoff += 0.02

    # Sort by pH value of max freq.
    ps = np.argsort(pH_argmaxs)
    state_strs_relevant = [state_strs_relevant[p] for p in ps]
    sfreqs_relevant = [sfreqs_relevant[p] for p in ps]
    mols_relevant = [mols_relevant[p] for p in ps]
    if verbose:
        print(f'Final N relevant states: {N_relevant_states} with cutoff {cutoff}')
    return N_relevant_states, state_strs_relevant, sfreqs_relevant, mols_relevant, sfreqs_not_relevant

def plot_relevant_states(mols_relevant,p):

    for mol in mols_relevant: tmp=AllChem.Compute2DCoords(mol)

    for mol in mols_relevant:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    
    img=MolsToGridImage(mols_relevant,molsPerRow=4,subImgSize=(150,150),legends=[x.GetProp("_Name") for x in mols_relevant],returnPNG=False,useSVG=True)

    img = img.replace('fill:#FFFFFF', 'fill:none')

    with open(f'{p.path_figs}/{p.name}_relevant_states.svg','w') as f:
        if p.notebook:
            f.write(img.data)
        else:
            f.write(img)

def plot_optimal_state(mol, p):
    tmp=AllChem.Compute2DCoords(mol)
    MolToFile(mol, f'{p.path_figs}/{p.name}_opti.svg', size=(800,630), imageType='svg')
    cairosvg.svg2pdf(url=f'{p.path_figs}/{p.name}_opti.svg',write_to=f'{p.path_figs}/{p.name}_opti.pdf')
    os.system(f'rm {p.path_figs}/{p.name}_opti.svg')

def compose_image(N_relevant_states,p):
    if N_relevant_states % 4 == 0:
        y = 350 + (N_relevant_states//4) * 150
    else:
        y = 350 + (N_relevant_states//4 + 1) * 150
    Figure(
        "600px", f"{y}px",
        SVG(f'{p.path_figs}/{p.name}_ph_scan.svg').move(30, 0),
        SVG(f'{p.path_figs}/{p.name}_relevant_states.svg').move(0, 350)
    ).save(f'{p.path_figs}/{p.name}_combined.svg')

    cairosvg.svg2pdf(url=f'{p.path_figs}/{p.name}_combined.svg',write_to=f'{p.path_figs}/{p.name}_combined.pdf')
    os.system(f'rm {p.path_figs}/{p.name}_ph_scan.svg')
    os.system(f'rm {p.path_figs}/{p.name}_relevant_states.svg')
    os.system(f'rm {p.path_figs}/{p.name}_combined.svg')