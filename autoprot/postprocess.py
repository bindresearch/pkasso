import numpy as np
import matplotlib.pyplot as plt

from svgutils.compose import Figure, SVG # type: ignore
# import cairosvg before anything in rdkit.Chem.Draw, breaks otherwise !!
import cairosvg  # type: ignore

from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Chem.Draw import MolToFile, MolsToGridImage

import copy
import os

def plot_pH_scan(
    name: str,
    indices: list[int],
    state_strs_relevant: list[str],
    sfreqs_relevant: list[np.ndarray],
    pHs: np.ndarray,
    net_charges: np.ndarray,
    sfreqs_not_relevant: list[np.ndarray],
    pkas_combined: dict[int, float],
    path: str = 'figures',
    verbose: bool = False,
    ) -> None:
    """ Plot scan of microstate frequencies for different pH values. """
    
    if len(pHs) == 1:
        style = 'o'
    else:
        style = '-'

    fsave = f'{path}/{name}_ph_scan.svg'
    cmap = plt.cm.get_cmap("Spectral_r")

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
            color_rb = 'tab:blue'
        else:
            color_rb = 'tab:red'
        ax[1].plot(pHs[x],net_charges[x],'o',color=color_rb,markersize=5)
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

def export_sdf(state_strs: list[str], mols_lib: dict[str, Mol], name: str, path_out: str) -> None:
    with Chem.SDWriter(f'{path_out}/{name}.sdf') as f:
        for e_idx, state_str in enumerate(state_strs):
            mol = mols_lib[state_str]
            mol_h = Chem.AddHs(mol)

            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
            if cid != 0:
                raise ValueError(f'{name}_{state_str} could not be embedded.')
            AllChem.UFFOptimizeMolecule(mol_h) # type: ignore
            mol_h.SetProp("_Name", f'{name}_{e_idx}')
            f.write(mol_h)

def export_csv(
    state_strs: list[str],
    smiles_lib: dict[str,str],
    sfreqs: np.ndarray,
    state_qs: dict[str, int],
    name: str,
    path_out: str,
    fout_csv: str,
    append: bool,
    ) -> None:
    """ Export csv with information about microstates at the given pH. """
    if append:
        action = 'a'
    else:
        action = 'w'
    with open(f'{path_out}/{fout_csv}',action) as f:
        if action == 'w':
            f.write(f'name,name_state,SMILES,frequency,charge\n')
        for e_idx, (state_str, sfreq) in enumerate(zip(state_strs, sfreqs)):
            smiles = smiles_lib[state_str]
            # Remove labels
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms(): # type: ignore
                atom.SetAtomMapNum(0)
            smiles = Chem.MolToSmiles(mol)
            f.write(f'{name},{name}_{e_idx},{smiles},{sfreq/np.sum(sfreqs):.5f},{state_qs[state_str]}\n')

def export_macro_pkas(pkas_combined: dict[int, float], name: str, path_out: str) -> None:
    """ Write macro pKas from pooled microstates. """
    for idx, (q, pka) in enumerate(pkas_combined.items()):
        print(f'pKa{idx+1} | {q+1} --> {q} | {pka:.3f}')
    with open(f'{path_out}/{name}_pkas.csv','w') as f:
        f.write('idx,q0,q1,pka\n')
        for idx, (q, pka) in enumerate(pkas_combined.items()):
            f.write(f'pKa{idx+1},{q},{q+1},{pka:.5f}\n')

def calc_relevant_states(
    state_freqs_all: dict[str, np.ndarray],
    mols_lib: dict[str, Mol],
    max_states: int = 18,
    verbose: bool = False,
    ) -> tuple[
        int,
        list[str],
        list[np.ndarray],
        list[Mol],
        list[np.ndarray]
    ]:
    """ Reduce number of states to max_states for plotting. """

    cutoff = 0.01
    tries = 0

    N_relevant_states = int(1e5)
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

def plot_relevant_states(mols_relevant: list[Mol], name: str, path_figs: str, notebook: bool) -> None:
    """ Plot rdkit molecules for relevant states together with state strings. """

    for mol in mols_relevant: tmp=AllChem.Compute2DCoords(mol) # type: ignore

    for mol in mols_relevant:
        for atom in mol.GetAtoms(): # type: ignore
            atom.SetAtomMapNum(0)
    
    img=MolsToGridImage(mols_relevant,molsPerRow=4,subImgSize=(150,150),legends=[x.GetProp("_Name") for x in mols_relevant],returnPNG=False,useSVG=True) # type: ignore

    img = img.replace('fill:#FFFFFF', 'fill:none')

    with open(f'{path_figs}/{name}_relevant_states.svg','w') as f:
        if notebook:
            f.write(img.data)
        else:
            f.write(img)

def plot_optimal_state(mol: Mol, name: str, path_figs: str) -> None:
    """ Plot state with highest frequency at pH_output. """

    tmp=AllChem.Compute2DCoords(mol) # type: ignore
    MolToFile(mol, f'{path_figs}/{name}_opti.svg', size=(800,630), imageType='svg') # type: ignore
    cairosvg.svg2pdf(url=f'{path_figs}/{name}_opti.svg',write_to=f'{path_figs}/{name}_opti.pdf')
    os.system(f'rm {path_figs}/{name}_opti.svg')

def compose_image(N_relevant_states: int, name: str, path_figs: str) -> None:
    """ Combine pH scan and plotted rdkit molecules. """
    if N_relevant_states % 4 == 0:
        y = 350 + (N_relevant_states//4) * 150
    else:
        y = 350 + (N_relevant_states//4 + 1) * 150
    Figure(
        "600px", f"{y}px",
        SVG(f'{path_figs}/{name}_ph_scan.svg').move(30, 0),
        SVG(f'{path_figs}/{name}_relevant_states.svg').move(0, 350)
    ).save(f'{path_figs}/{name}_combined.svg')

    cairosvg.svg2pdf(url=f'{path_figs}/{name}_combined.svg',write_to=f'{path_figs}/{name}_combined.pdf')
    os.system(f'rm {path_figs}/{name}_ph_scan.svg')
    os.system(f'rm {path_figs}/{name}_relevant_states.svg')
    os.system(f'rm {path_figs}/{name}_combined.svg')