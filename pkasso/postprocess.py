"""Postprocessing utilities for pKasso outputs."""

import copy
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

### import cairosvg before anything in rdkit.Chem.Draw, svgutils breaks otherwise !!
import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure as Figure_plt
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Chem.Draw import MolsToGridImage, MolDrawOptions
from svgutils.compose import SVG, Figure

from .utils import is_jupyter, state_str_to_q

logger = logging.getLogger(__name__)

def draw_mols(mols: list[Mol], subImgSize: tuple[int, int] = (250, 200), max_cols=4, show_probability=True) -> Any:
    opts = MolDrawOptions()
    opts.backgroundColour = (1, 1, 1, 1) # type: ignore

    if show_probability:
        legends = [f'{x.GetProp("_Name")}\n{float(x.GetProp("Probability"))*100:.2f}'+r'%' for x in mols]
    else:
        legends = [f'{x.GetProp("_Name")}' for x in mols]

    img = MolsToGridImage( # type: ignore
    mols,
    molsPerRow=min(max_cols,len(mols)),
    subImgSize=subImgSize,
    legends=legends,
    returnPNG=False,
    useSVG=True,
    drawOptions=opts,
    )
    return img

def save_sdf(mols: list[Mol], file: Path) -> None:
    """ Save embedded and optimized mols to sdf"""
    with Chem.SDWriter(file) as f:
        for mol in mols:
            mol_3d = copy.deepcopy(mol)
            mol_h = Chem.AddHs(mol_3d)
            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
            if cid != 0:
                raise ValueError(f'{mol.GetProp("_Name")} could not be embedded.')
            AllChem.UFFOptimizeMolecule(mol_h) # type: ignore
            f.write(mol_h)

@dataclass
class Microstate:
    """ Single microstate class for output. """

    name: str
    name_state: str
    mol: Mol
    smiles: str
    freq: float
    q: int

@dataclass
class Molecule:
    """ Molecule class storing the microstate output from pKasso for output. """

    name: str
    microstates: list[Microstate]
    
    def __post_init__(self) -> None:
        """ Add attributes to list properties of all microstates of a molecule. """

        self.smiles: list[str] = [m.smiles for m in self.microstates]
        self.mols: list[Mol] = [m.mol for m in self.microstates]
        self.freqs: list[float] = [m.freq for m in self.microstates]
        self.qs: list[int] = [m.q for m in self.microstates]

def combine_results(
        name: str,
        state_strs: list[str],
        state_freqs: list[float],
        mols_lib: dict[str, Mol],
        state_qs: dict[str, int],
) -> Molecule:
    """ Clean up smiles and mols for output. """

    microstates: list[Microstate] = []

    for e_idx, (state_str, sfreq) in enumerate(zip(state_strs, state_freqs)):
        name_state = f'{name}_{e_idx}'
        mol = copy.deepcopy(mols_lib[state_str])
        mol.SetProp("_Name", name_state)
        mol.SetProp("Probability", f'{sfreq}')
        mol.SetProp('state_str', state_str)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        _ = AllChem.Compute2DCoords(mol) # type: ignore
        smiles = Chem.MolToSmiles(mol)
        mol.SetProp('SMILES', smiles)
        sfreq_out = sfreq/np.sum(state_freqs)
        q = state_qs[state_str]
        mol.SetProp('net_charge', f'{q:.5f}')

        res = Microstate(name, name_state, mol, smiles, float(sfreq_out), q)
        microstates.append(res)

    molecule = Molecule(name, microstates)
    return molecule

@dataclass
class Scan:
    """ 
    Store the output of a pH scan.
    Includes postprocessing methods for plotting.
    """

    name: str
    indices: list[int]
    state_strs_relevant: list[str]
    mols_relevant: list[Mol]
    sfreqs_relevant: list[NDArray[np.float64]]
    pHs: NDArray[np.float64]
    net_charges: NDArray[np.float64]
    sfreqs_not_relevant: list[NDArray[np.float64]]
    pkas_macro: dict[int, float]

    def __post_init__(self) -> None:
        self.state_strs_conv = [state_str_to_q(state_str) for state_str in self.state_strs_relevant]

    def export_macro_pkas(self, file: Path) -> None:
        """ Write macro pKas from pooled microstates. """
        with open(file,'w') as f:
            f.write('idx,q0,q1,pka\n')
            for idx, (q, pka) in enumerate(self.pkas_macro.items()):
                f.write(f'pKa{idx+1},{q},{q+1},{pka:.5f}\n')

    def print_macro_pkas(self) -> None:
        """ Print macro pKa values. """
        print('Macro-pKa values:')
        for idx, (q, pka) in enumerate(self.pkas_macro.items()):
            print(f'pKa{idx+1} | {q+1} --> {q} | {pka:.3f}')

    def plot_mols(self, size_x: int = 200, size_y: int = 175, molsPerRow: int = 4) -> Any:
        """ Plot rdkit molecules for relevant states together with state strings. 
        
        Returns IPython.core.display.SVG when called from notebook
        or str when called from script
        
        """

        opts = MolDrawOptions()
        opts.backgroundColour = (1, 1, 1, 1) # type: ignore

        for mol in self.mols_relevant:
            _ = AllChem.Compute2DCoords(mol) # type: ignore
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        
        fig_mols = MolsToGridImage(
            self.mols_relevant,
            molsPerRow=molsPerRow,
            subImgSize=(size_x,size_y),
            legends=[x.GetProp("_Name") for x in self.mols_relevant],
            returnPNG=False,
            useSVG=True,
            drawOptions=opts,
        ) # type: ignore
        return fig_mols

    def plot_scan(
        self,
        highlight_idx = 0,
        ) -> Figure_plt:
        """ Plot scan of microstate frequencies for different pH values. """
        
        # print(highlight_idx)

        if len(self.pHs) == 1:
            style = 'o'
        else:
            style = '-'

        cmap = plt.cm.get_cmap("Spectral_r")

        px = 1/plt.rcParams['figure.dpi']

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
            
            alpha = 1.0
            lw = 1.5
            if highlight_idx > 0:
                if highlight_idx-1 == idx:
                    lw = 2.0
                else:
                    color = 'gray'
                    alpha = 0.3
                    lw = 1.0
            ax[0].plot(self.pHs,sfreq*100,style,label=state_str,color=color,alpha=alpha,lw=lw)
        if len(self.state_strs_conv) > 8:
            ax[0].legend(ncol=2,fontsize=8)
        elif len(self.state_strs_conv) > 1:
            ax[0].legend(ncol=1,fontsize=10)

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
    
    def export_scan(self, file: Path, fig_scan: Figure_plt, fig_mols: Any,
                    size_y: float = 150.) -> None:
        """ Combine pH scan and molecule drawings and save to file. """

        N_relevant_states = len(self.state_strs_relevant)

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=True) as f:
            fig_scan.savefig(f.name, transparent=True)
            f.flush()
            svg_scan = SVG(f.name)

        img_data: str = ''
        if is_jupyter():
            img_data = fig_mols.data
        else:
            img_data = fig_mols

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=True) as f:
            f.write(img_data)
            f.flush()
            svg_mols = SVG(f.name)

        self.compose_image(svg_scan, svg_mols, file, N_relevant_states, size_y = size_y)

    @staticmethod
    def compose_image(svg_scan: SVG, svg_mols: SVG, file: Path, N_relevant_states: int,
                      size_y: float = 150.) -> None:
        """ Combine SVG objects of pH scan and rdkit molecule drawings. """

        if N_relevant_states % 4 == 0:
            y = 420 + (N_relevant_states//4) * size_y
        else:
            y = 420 + (N_relevant_states//4 + 1) * size_y

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=True) as f:
            Figure(
                "600px", f"{y}px",
                svg_scan.move(0, 0),
                svg_mols.move(0, 420),
            ).save(f.name)

            cairosvg.svg2pdf(url=f.name, write_to=os.fspath(file))

    def save_sdf(self, file: Path) -> None:
        """ 
        Write sdf file with all relevant mols, optimized geometry with rdkit.
        Includes explicit hydrogens.
        """

        with Chem.SDWriter(file) as f:
            for mol in self.mols_relevant:
                mol_3d = copy.deepcopy(mol)
                mol_h = Chem.AddHs(mol_3d)
                cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
                if cid != 0:
                    raise ValueError(f'{mol.GetProp("_Name")} could not be embedded.')
                AllChem.UFFOptimizeMolecule(mol_h) # type: ignore
                f.write(mol_h)
