"""Utility helpers for protonation state processing."""

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.rdchem import Atom, Mol


def pack_vec(state_vec: NDArray[np.int64]) -> str:
    """ Pack vector into string. """

    state_str = "".join([str(x) for x in state_vec])
    return state_str

def unpack_vec(state_str: str) -> NDArray[np.int64]:
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

def calc_qs_all(state_vecs: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
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

def sort_string(string: str, ps: NDArray[np.int64]) -> str:
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
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config # type: ignore
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

def read_smi(smi: Path) -> dict[str, str]:
    """Parse input .smi files"""

    # smiles_batch = []
    # names_batch = []
    batch_dict: dict[str, str] = {}

    with open(smi,'r') as f:
        for line in f.readlines():
            spl = line.split()
            batch_dict[spl[1]] = spl[0]
            # smiles_batch.append(spl[0])
            # names_batch.append(spl[1])
    return batch_dict