
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom
from numpy.typing import NDArray

from typing import Iterable, cast
import copy


def pack_vec(state_vec: np.ndarray) -> str:
    """ Pack vector into string. """
    state_str = "".join([str(x) for x in state_vec])
    return state_str

def unpack_vec(state_str: str) -> np.ndarray:
    """ Unpack string into vector. """
    state_vec = np.array([int(s) for s in state_str],dtype=int)
    return state_vec

def calc_state_strs(state_vecs: list[NDArray[np.int64]]) -> list[str]:
    """ Calc state strings from vectors. """
    state_strs = []
    for state_vec in state_vecs:
        state_str = pack_vec(state_vec)
        state_strs.append(state_str)
    return state_strs

def calc_qs_all(state_vecs: list[np.ndarray]) -> list[np.ndarray]:
    """ Convert state vectors into a vector of charges. """
    qs_all = []
    for state_vec in state_vecs:
        qs = state_vec - 1
        qs_all.append(qs)
    return qs_all

def get_atom_with_map_idx(mol: Mol, map_idx: int) -> Atom | None:
    """ Find atom of rdkit Mol object with specific map index. """
    for atom in cast(list[Atom], mol.GetAtoms()): # type: ignore
        if atom.GetAtomMapNum() == map_idx:
            return atom
    return None

def sort_string(string: str, ps: np.ndarray) -> str:
    """ Sort string by custom indices ps. """

    s = list(string)
    s = [s[p] for p in ps]
    s_out = "".join(s)
    return s_out

def pack_indices(indices: list[int]) -> str:
    """ Convert list of indices into comma-separated string. """
    indices_str = ''
    for id in indices:
        indices_str += f'{id},'
    indices_str = indices_str[:-1] # remove last comma
    return indices_str

def is_jupyter() -> bool:
    """ Check if a jupyter notebook/lab is run."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except ImportError:
        return False

def state_str_to_q(state_str: str) -> str:
    """ Convert state_str (0, 1, 2) to 
    string of charges (-, 0, +) """

    state_str_to_q_dict = {
        '0' : '-',
        '1' : '0',
        '2' : '+',
    }

    q = ''
    for s in state_str:
        q += state_str_to_q_dict[s]
    return q

#### INPUT / OUTPUT ####

def read_smi(smi):
    smiles_batch = []
    names_batch = []

    with open(smi,'r') as f:
        for line in f.readlines():
            spl = line.split()
            smiles_batch.append(spl[0])
            names_batch.append(spl[1])
    return names_batch, smiles_batch

# def export_relevant_states(img):
#     if is_jupyter():
#         img_data: str = img.data
#     else:
#         img_data: str = img
#     img_data = img_data.replace('fill:#FFFFFF', 'fill:none')
#     with open(f'tmp_states.svg','w') as f:
#         f.write(img_data)

# def export_pH_scan(fig):
    # fig.savefig(f'tmp_plt.svg', transparent=True)

# def compose_image(N_relevant_states: int, file: str | None = None) -> None:# name: str, path_out: str) -> None:
#     """ Combine pH scan and plotted rdkit molecules. """
#     if N_relevant_states % 4 == 0:
#         y = 350 + (N_relevant_states//4) * 150
#     else:
#         y = 350 + (N_relevant_states//4 + 1) * 150
#     Figure(
#         "600px", f"{y}px",
#         SVG(f'tmp_plt.svg').move(30, 0),
#         SVG(f'tmp_states.svg').move(0, 350)
#     ).save(f'tmp_combined.svg')

#     cairosvg.svg2pdf(url=f'tmp_combined.csv',write_to=f'{file}')
#     os.system(f'rm tmp_plt.svg')
#     os.system(f'rm tmp_states.svg')
#     os.system(f'rm tmp_combined.svg')