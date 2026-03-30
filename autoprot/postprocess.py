from svgutils.compose import Figure, SVG # type: ignore
import svgutils.transform as sg
# import cairosvg before anything in rdkit.Chem.Draw, breaks otherwise !!
import cairosvg  # type: ignore

import numpy as np
import matplotlib.pyplot as plt

from .utils import *

from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Chem.Draw import MolToFile, MolsToGridImage

import pandas as pd

import copy
import os
import io
import tempfile

from dataclasses import dataclass

from typing import Any

@dataclass
class Microstate:
    name: str
    name_state: str
    mol: Mol
    smiles: str
    freq: float
    q: int

@dataclass
class Molecule:
    name: str
    microstates: list[Microstate]
    
    def __post_init__(self):
        self.smiles = [m.smiles for m in self.microstates]
        self.mols = [m.mol for m in self.microstates]
        self.freqs = [m.freq for m in self.microstates]
        self.qs = [m.q for m in self.microstates]

    def draw(self):
        mols = [state.mol for state in self.microstates]
        img = MolsToGridImage(
            mols,molsPerRow=len(mols),subImgSize=(250,200),
            legends=[x.GetProp("_Name") for x in mols],
            returnPNG=False,useSVG=True
            ) # type: ignore
        return img
    
    def save(self, file: str) -> None:
        """ 
        Write sdf file with all relevant mols, optimized geometry with rdkit.
        Includes explicit hydrogens.
        """
        with Chem.SDWriter(f'{file}') as f:
            for mol in self.mols:
                mol_3d = copy.deepcopy(mol)
                mol_h = Chem.AddHs(mol_3d)
                cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
                if cid != 0:
                    raise ValueError(f'{mol.GetProp("_Name")} could not be embedded.')
                AllChem.UFFOptimizeMolecule(mol_h) # type: ignore
                f.write(mol_h)

def combine_results(name: str, state_strs: list[str], state_freqs: list[float], mols_lib: dict[str, Mol], state_qs: dict[str, int]
) -> Molecule:
    """ Clean up smiles and mols for output. """

    microstates: list[Microstate] = []

    for e_idx, (state_str, sfreq) in enumerate(zip(state_strs, state_freqs)):
        name_state = f'{name}_{e_idx}'
        mol = copy.deepcopy(mols_lib[state_str])
        mol.SetProp("_Name", name_state)
        mol.SetProp("Frequency", f'{sfreq}')
        for atom in mol.GetAtoms(): # type: ignore
            atom.SetAtomMapNum(0)
        tmp=AllChem.Compute2DCoords(mol)
        smiles = Chem.MolToSmiles(mol)
        sfreq_out = sfreq/np.sum(state_freqs)
        q = state_qs[state_str]

        res = Microstate(name, name_state, mol, smiles, float(sfreq_out), q)
        microstates.append(res)

    molecule = Molecule(name, microstates)
    return molecule

@dataclass
class Batch:
    molecules: dict[str, Molecule]

    def to_pandas(self):
        df = pd.DataFrame()
        for name, molecule in self.molecules.items():
            for m in molecule.microstates:
                df.loc[m.name_state,'name'] = m.name
                df.loc[m.name_state,'SMILES'] = m.smiles
                df.loc[m.name_state,'frequency'] = m.freq
                df.loc[m.name_state,'q'] = m.q
        df.index.rename('name_state', inplace=True)
        return df

@dataclass
class Scan:
    name: str
    indices: list[int]
    state_strs_relevant: list[str]
    mols_relevant: list[Mol]
    sfreqs_relevant: list[np.ndarray]
    pHs: np.ndarray
    net_charges: np.ndarray
    sfreqs_not_relevant: list[np.ndarray]
    pkas_macro: dict[int, float]

    def __post_init__(self):
        self.state_strs_conv = [state_str_to_q(state_str) for state_str in self.state_strs_relevant]

    def export_macro_pkas(self, file) -> None:
        """ Write macro pKas from pooled microstates. """
        with open(f'{file}','w') as f:
            f.write('idx,q0,q1,pka\n')
            for idx, (q, pka) in enumerate(self.pkas_macro.items()):
                f.write(f'pKa{idx+1},{q},{q+1},{pka:.5f}\n')

    def print_macro_pkas(self) -> None:
        print(f'Macro-pKa values:')
        for idx, (q, pka) in enumerate(self.pkas_macro.items()):
            print(f'pKa{idx+1} | {q+1} --> {q} | {pka:.3f}')

    def plot_mols(self, size_x: float = 200, size_y: float = 175) -> None:# name: str, path_out: str) -> None:
        """ Plot rdkit molecules for relevant states together with state strings. """

        for mol in self.mols_relevant: tmp=AllChem.Compute2DCoords(mol) # type: ignore
        for mol in self.mols_relevant:
            for atom in mol.GetAtoms(): # type: ignore
                atom.SetAtomMapNum(0)
        
        fig_mols = MolsToGridImage(self.mols_relevant,molsPerRow=4,subImgSize=(size_x,size_y), legends=[state_str_to_q(x.GetProp("_Name")) for x in self.mols_relevant],returnPNG=False,useSVG=True) # type: ignore
        return fig_mols

    # @staticmethod
    # def export_mols(file: str, img: Any) -> None: # FIX TYPING
    #     if is_jupyter():
    #         img_data: str = img.data
    #     else:
    #         img_data: str = img
    #     img_data = img_data.replace('fill:#FFFFFF', 'fill:none')
    #     with open(file,'w') as f:
    #         f.write(img_data)

    def plot_scan(self,
        ) -> None:
        """ Plot scan of microstate frequencies for different pH values. """
        
        if len(self.pHs) == 1:
            style = 'o'
        else:
            style = '-'

        # fsave = f'{path}/{name}_ph_scan.svg'
        cmap = plt.cm.get_cmap("Spectral_r")

        px = 1/plt.rcParams['figure.dpi']

        # fig_scan, ax = plt.subplots(2,1,figsize=(700*px,500*px),height_ratios=[0.6,0.4])
        fig_scan, ax = plt.subplots(2,1,figsize=(820*px,600*px),height_ratios=[0.6,0.4])

        for idx, sfreq in enumerate(self.sfreqs_not_relevant):
            ax[0].plot(self.pHs,sfreq*100,style,color='gray',lw=1.,alpha=0.3)

        for idx, (state_str, sfreq) in enumerate(zip(self.state_strs_conv,self.sfreqs_relevant)):
            if len(self.state_strs_conv) > 1:
                if len(self.state_strs_conv) == 3 and idx == 1:
                    color = cmap((idx-0.3)/(len(self.state_strs_conv)-1)) # avoid invisible yellow
                else:
                    color = cmap(idx/(len(self.state_strs_conv)-1))
            else:
                color = cmap(0)
            ax[0].plot(self.pHs,sfreq*100,style,label=state_str,color=color)
        if len(self.state_strs_conv) > 8:
            ax[0].legend(ncol=2,fontsize=8)
        elif len(self.state_strs_conv) > 1:
            ax[0].legend(ncol=1,fontsize=10)

        # ax[0].set(xlabel='pH',ylabel='Probability [%]')
        ax[0].set_xlabel('pH', fontsize=12)
        ax[0].set_ylabel('Probability [%]', fontsize=12)
        ax[0].grid(alpha=0.3)

        ax[1].plot(self.pHs,self.net_charges,style,color='black')

        for idx, (q, pka) in enumerate(self.pkas_macro.items()):
            x = np.argmin(np.abs(self.pHs-pka))
            if q+1 > 0:
                color_rb = 'tab:blue'
            else:
                color_rb = 'tab:red'
            ax[1].plot(self.pHs[x],self.net_charges[x],'o',color=color_rb,markersize=5)
            ax[1].text(self.pHs[x]+0.1,self.net_charges[x]+0.05,f'{pka:.2f}',fontsize=12)

        ax[1].set_xlabel('pH', fontsize=12)
        ax[1].set_ylabel('Net charge', fontsize=12)

        ax[1].grid(alpha=0.3)

        if len(self.pHs) > 1:
            for idx in range(2):
                ax[idx].set(xlim=(self.pHs[0],self.pHs[-1]))
                ax[idx].set_xticks(np.arange(self.pHs[0],self.pHs[-1]+0.001,1))
                ax[idx].tick_params(axis='both', which='major', labelsize=12)

        ax[0].set_title(self.name)

        fig_scan.tight_layout()
        plt.close(fig_scan)
        return fig_scan
    
    def export_scan(self, file: str, fig_scan: Any, fig_mols: Any,
                    size_x: float = 150., size_y: float = 150.): # Fix Typing

        N_relevant_states = len(self.state_strs_relevant)

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=True) as f:
            fig_scan.savefig(f.name, transparent=True)
            f.flush()
            svg_scan = SVG(f.name)

        if is_jupyter():
            img_data: str = fig_mols.data
        else:
            img_data: str = fig_mols
        img_data = img_data.replace('fill:#FFFFFF', 'fill:none')

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=True) as f:
            f.write(img_data)
            f.flush()
            svg_mols = SVG(f.name)

        self.compose_image(svg_scan, svg_mols, file, N_relevant_states, size_y = size_y)

    @staticmethod
    def compose_image(svg_scan: SVG, svg_mols: SVG, file: str, N_relevant_states: int,
                      size_y: float = 150.):
        if N_relevant_states % 4 == 0:
            # y = 350 + (N_relevant_states//4) * size_y
            y = 420 + (N_relevant_states//4) * size_y
        else:
            # y = 350 + (N_relevant_states//4 + 1) * size_y
            y = 420 + (N_relevant_states//4 + 1) * size_y

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=True) as f:
            Figure(
                "600px", f"{y}px",
                svg_scan.move(0, 0),
                # svg_mols.move(0, 350),
                svg_mols.move(0, 420),
            ).save(f.name)#f'{file[:-4]}.svg')

            # cairosvg.svg2pdf(url=f'{file[:-4]}.svg',write_to=f'{file}')
            cairosvg.svg2pdf(url=f.name,write_to=f'{file}')
        # os.system(f'rm {file[:-4]}.svg')

    def save_sdf(self, file: str) -> None:
        """ 
        Write sdf file with all relevant mols, optimized geometry with rdkit.
        Includes explicit hydrogens.
        """
        with Chem.SDWriter(f'{file}') as f:
            for mol in self.mols_relevant:
                mol_3d = copy.deepcopy(mol)
                mol_h = Chem.AddHs(mol_3d)
                cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
                if cid != 0:
                    raise ValueError(f'{mol.GetProp("_Name")} could not be embedded.')
                AllChem.UFFOptimizeMolecule(mol_h) # type: ignore
                f.write(mol_h)

# def plot_optimal_state(mol: Mol, name: str, path_out: str) -> None:
#     """ Plot state with highest frequency at pH. """

#     tmp=AllChem.Compute2DCoords(mol) # type: ignore
#     MolToFile(mol, f'{path_out}/{name}_opti.svg', size=(800,630), imageType='svg') # type: ignore
#     cairosvg.svg2pdf(url=f'{path_out}/{name}_opti.svg',write_to=f'{path_out}/{name}_opti.pdf')
#     os.system(f'rm {path_out}/{name}_opti.svg')

    # Figure(
    #     "600px", f"{y}px",
    #     SVG(f'{path_out}/{name}_ph_scan.svg').move(30, 0),
    #     SVG(f'{path_out}/{name}_relevant_states.svg').move(0, 350)
    # ).save(f'{path_out}/{name}_combined.svg')

    # cairosvg.svg2pdf(url=f'{path_out}/{name}_combined.svg',write_to=f'{path_out}/{name}_scan.pdf')
    # os.system(f'rm {path_out}/{name}_ph_scan.svg')
    # os.system(f'rm {path_out}/{name}_relevant_states.svg')
    # os.system(f'rm {path_out}/{name}_combined.svg')