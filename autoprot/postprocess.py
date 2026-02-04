import numpy as np
import matplotlib.pyplot as plt

from svgutils.compose import Figure, SVG
import cairosvg # import before anything in rdkit.Chem.Draw, breaks otherwise !!

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToFile, MolsToGridImage

import os

# def get_relevant_states(state_strs, state_freqs_all, mols_lib, cutoff=0.05):
#     state_strs_relevant = []
#     sfreqs = state_freqs_all.T
#     sfreqs_relevant = []
#     mols_relevant = []
    
#     for idx, (state_str, sfreq) in enumerate(zip(state_strs,sfreqs)):
#         if np.max(sfreq) > cutoff:
#             state_strs_relevant.append(state_str)
#             sfreqs_relevant.append(sfreq)
#             mols_relevant.append(mols_lib[state_str])

#     return state_strs_relevant, sfreqs_relevant, mols_relevant

def plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, sfreqs_not_relevant, cmap=plt.cm.Spectral,
                 path='figures'):
    cmap = plt.cm.Spectral_r

    fsave = f'{path}/{name}_ph_scan.svg'

    print(f'Indices: {indices}')
    px = 1/plt.rcParams['figure.dpi']

    fig, ax = plt.subplots(2,1,figsize=(700*px,500*px),height_ratios=[0.6,0.4])

    for idx, sfreq in enumerate(sfreqs_not_relevant):
        ax[0].plot(pHs,sfreq*100,color='gray',lw=1.,alpha=0.3)

    for idx, (state_str, sfreq) in enumerate(zip(state_strs_relevant,sfreqs_relevant)):
        if len(state_strs_relevant) > 1:
            color = cmap(idx/(len(state_strs_relevant)-1))
        else:
            color = cmap(0)
        ax[0].plot(pHs,sfreq*100,label=state_str,color=color)
    if len(state_strs_relevant) > 10:
        ax[0].legend(ncol=2,fontsize=6)
    else:
        ax[0].legend(ncol=1,fontsize=8)
    ax[0].set(xlabel='pH',ylabel='Distribution %')
    ax[0].set(xlim=(pHs[0],pHs[-1]))
    ax[0].set_title(name)
    ax[0].set_xticks(np.arange(pHs[0],pHs[-1],1))
    ax[0].grid(alpha=0.3)

    ax[1].plot(pHs,net_charges,color='black')
    ax[1].set(xlabel='pH', ylabel='Net charge')
    ax[1].set(xlim=(pHs[0],pHs[-1]))
    ax[1].grid(alpha=0.3)
    ax[1].set_xticks(np.arange(pHs[0],pHs[-1],1))

    fig.tight_layout()
    if fsave != '':
        fig.savefig(fsave)
    plt.close()
    # return fig

def export_sdf(name,mol_h,path='output'):
    AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)
    AllChem.UFFOptimizeMolecule(mol_h)
    with Chem.SDWriter(f'{path}/{name}.sdf') as f:
        f.write(mol_h)

def export_smi(name,smiles,path='output'):
    with open(f'{path}/{name}.smi','w') as f:
        f.write(f'{smiles} {name}\n')

def plot_relevant_states(name, mols_relevant,path='figures'):

    for mol in mols_relevant: tmp=AllChem.Compute2DCoords(mol)
    
    img=MolsToGridImage(mols_relevant,molsPerRow=4,subImgSize=(150,150),legends=[x.GetProp("_Name") for x in mols_relevant],returnPNG=False,useSVG=True)

    with open(f'{path}/{name}_relevant_states.svg','w') as f:
        # f.write(img.data)
        f.write(img)

def plot_optimal_state(name, mol, path='figures'):
    tmp=AllChem.Compute2DCoords(mol)
    MolToFile(mol, f'{path}/{name}_opti.svg', size=(800,630), imageType='svg')
    cairosvg.svg2pdf(url=f'{path}/{name}_opti.svg',write_to=f'{path}/{name}_opti.pdf')
    os.system(f'rm {path}/{name}_opti.svg')

def compose_image(name,N_relevant_states,path='figures'):
    if N_relevant_states % 4 == 0:
        y = 350 + (N_relevant_states//4) * 150
    else:
        y = 350 + (N_relevant_states//4 + 1) * 150
    Figure(
        "600px", f"{y}px",
        SVG(f'{path}/{name}_ph_scan.svg').move(30, 0),
        SVG(f'{path}/{name}_relevant_states.svg').move(0, 350)
    ).save(f'{path}/{name}_combined.svg')

    cairosvg.svg2pdf(url=f'{path}/{name}_combined.svg',write_to=f'{path}/{name}_combined.pdf')
    os.system(f'rm {path}/{name}_ph_scan.svg')
    os.system(f'rm {path}/{name}_relevant_states.svg')
    os.system(f'rm {path}/{name}_combined.svg')
