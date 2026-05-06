"""Special-case handling for protonation state generation."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from collections import deque

from .transitions import calc_freqs_from_states, calc_state_diffs
from .utils import unpack_vec

logger = logging.getLogger(__name__)

def match_pattern(mol: Mol, pattern: Mol) -> tuple[bool, list[list[int]]]:
    """ Match pattern in rdkit molecule.
    
    Parameters:
    -----------
    mol
        Rdkit molecule
    pattern
        Pattern mol to match.

    Returns
    -------
    found
        Boolean returning if at least one match was found.
    matches
        Atom indices for each match.
    """

    found = mol.HasSubstructMatch(pattern)
    matches = mol.GetSubstructMatches(pattern)
    return found, matches

def find_charged(mol: Mol) -> list[int]:
    """
    Find map indices that could not be neutralized during preprocessing.
    These are removed from consideration.
    """

    mol_h = Chem.rdmolops.AddHs(mol)
    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()]) # type: ignore

    charged_indices = []
    for at_idx, q in enumerate(q0s):
        atom = mol_h.GetAtomWithIdx(at_idx) 
        map_idx = atom.GetAtomMapNum()

        if q != 0:
            logger.info(f'Input molecule is charged at idx {at_idx}, map_idx {map_idx}!')
            charged_indices.append(map_idx)

    return charged_indices

def add_exclusions(mol: Mol) -> tuple[list[int], list[int]]:
    """ 
    Exclusions act on the q_options level. Exclusions are removed from consideration
    for protonation/deprotonation. This is specified separately for acids and bases.
    For example, only a protonation event could be excluded for a given index,
    but not the deprotonation event.
    This does not affect the indices or cluster splitting/
    
    Parameters
    ----------
    mol
        Input molecule with atom mapping numbers used to track sites.

    Returns
    -------
    exclude_base_indices
        Atom map indices for which protonation (base behavior) should
        be excluded.
    exclude_acid_indices
        Atom map indices for which deprotonation (acid behavior) should
        be excluded.
    """

    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()]) # type: ignore

    exclude_acid_indices: list[int] = []
    exclude_base_indices: list[int] = []
    mol_h = Chem.rdmolops.AddHs(mol)

    pattern_carbonyl = Chem.MolFromSmarts("NC(=O)")
    found_carbonyl, matches_carbonyl = match_pattern(mol,pattern_carbonyl)
    pattern_imine = Chem.MolFromSmarts("NC(=N)")
    found_imine, matches_imine = match_pattern(mol,pattern_imine)
    pattern_sulfonamide = Chem.MolFromSmarts("NS(=O)(=O)")
    found_sulfonamide, matches_sulfonamide = match_pattern(mol, pattern_sulfonamide)
    pattern_diphenylamine = Chem.MolFromSmarts('N(c)c')
    found_diphenylamine, matches_diphenylamine = match_pattern(mol, pattern_diphenylamine)
    pattern_Ncnn = Chem.MolFromSmarts('Nc(n)n')
    found_Ncnn, matches_Ncnn = match_pattern(mol, pattern_Ncnn)
    pattern_Nccn = Chem.MolFromSmarts('Nc(c)n')
    found_Nccn, matches_Nccn = match_pattern(mol, pattern_Nccn)
    pattern_nnn = Chem.MolFromSmarts('nnn')
    found_nnn, matches_nnn = match_pattern(mol, pattern_nnn)
    pattern_ncnn = Chem.MolFromSmarts('ncnn')
    found_ncnn, matches_ncnn = match_pattern(mol, pattern_ncnn)
    pattern_nco = Chem.MolFromSmarts('nco')
    found_nco, matches_nco = match_pattern(mol, pattern_nco)
    pattern_cNO = Chem.MolFromSmarts('C=NO')
    _, matches_cNO = match_pattern(mol, pattern_cNO)
    pattern_NNC = Chem.MolFromSmarts('N-N=C')
    _, matches_NNC = match_pattern(mol, pattern_NNC)
    #
    pattern_ONphos = Chem.MolFromSmarts('OC=NP(=O)(O)O')
    _, matches_ONphos = match_pattern(mol, pattern_ONphos)
    pattern_ONO = Chem.MolFromSmarts('[O]-[N+]([O-])')
    _, matches_ONO = match_pattern(mol, pattern_ONO)

    for at_idx, q in enumerate(q0s):
        atom = mol_h.GetAtomWithIdx(at_idx)
        map_idx = atom.GetAtomMapNum()

        if q == 0:
            # atom = mol_h.GetAtomWithIdx(at_idx)
            if atom.GetSymbol() == 'O':
                for match in matches_ONphos:
                    if (atom.GetIdx() in match):
                        correct_O = False
                        neighbors = atom.GetNeighbors()
                        for nbr in neighbors:
                            if nbr.GetAtomicNum() == 6:
                                correct_O = True # O=CN part of the match
                        if correct_O:
                            if map_idx not in exclude_acid_indices:
                                exclude_acid_indices.append(map_idx)
                for match in matches_ONO:
                    if (atom.GetIdx() in match):
                        if map_idx not in exclude_acid_indices:
                            exclude_acid_indices.append(map_idx)


            if atom.GetSymbol() == 'N':
                if atom.GetIsAromatic():
                    if atom.GetDegree() == 3: # arom. N with lone pair needed for ring
                        exclude_base_indices.append(map_idx)
                    for matches in [matches_nnn, matches_ncnn, matches_nco]:
                        for match in matches:
                            if atom.GetIdx() in match:
                                if map_idx not in exclude_base_indices:
                                    exclude_base_indices.append(map_idx)
                else:
                    for match in matches_carbonyl: # ...N-C(=O)...
                        if atom.GetIdx() in match:
                            if map_idx not in exclude_base_indices:
                                exclude_base_indices.append(map_idx)
                            logger.debug('Excluding N next to carbonyl as base')
                            logger.debug(at_idx, map_idx)
                    for match in matches_imine: # ...N-C(=N)...
                        if atom.GetIdx() in match:
                            accept = True
                            for bond in atom.GetBonds(): # Find the correct of the two Ns
                                if bond.GetBondType() == Chem.BondType.DOUBLE:
                                    accept = False
                            if accept:
                                if map_idx not in exclude_base_indices:
                                    exclude_base_indices.append(map_idx)
                                logger.debug('Excluding N next to imine as base')
                                logger.debug(at_idx, map_idx)
                    for match in matches_sulfonamide:
                        if atom.GetIdx() in match:
                            if map_idx not in exclude_base_indices:
                                exclude_base_indices.append(map_idx)
                            logger.debug('Excluding N next to sulfonamide as base')
                            logger.debug(at_idx, map_idx)
                    for matches in [matches_diphenylamine, matches_Ncnn, matches_Nccn, matches_cNO, matches_NNC]:
                        for match in matches:
                            if atom.GetIdx() in match:
                                if map_idx not in exclude_base_indices:
                                    exclude_base_indices.append(map_idx)
                    # for match in matches_Ncnn:
                    #     if atom.GetIdx() in match:
                    #         if map_idx not in exclude_base_indices:
                    #             exclude_base_indices.append(map_idx)
                    # for match in matches_Nccn:
                    #     if atom.GetIdx() in match:
                    #         if map_idx not in exclude_base_indices:
                    #             exclude_base_indices.append(map_idx)

    exclude_base_indices = sorted(exclude_base_indices)
    exclude_acid_indices = sorted(exclude_acid_indices)

    return exclude_base_indices, exclude_acid_indices
    
def add_exceptions(mol: Mol) -> tuple[list[int], dict[int, list[int]]]:
    """
    Identify indices that should be treated as separate protonation clusters.

    Exceptions decouple specific sites from the normal clustering procedure. 
    This is used for cases such as special functional groups
    (e.g., phosphates) that require dedicated treatment.

    Returns
    -------
    except_indices
        Atom map indices that should be removed from normal clustering.
    phosphate_groups
        Mapping from phosphate P atom map indices to their protonable OH
        atom map indices.
    """

    except_indices = []

    NphenNOO_indices = []
    # n_poor_arom_indices = []

    pattern_NphenNOO = Chem.MolFromSmarts("Ncccc([N+](=O)[O-])")
    found_NphenNOO, matches_NphenNOO = match_pattern(mol,pattern_NphenNOO)

    # patterns_n_poor_arom = [Chem.MolFromSmarts('nnn'), Chem.MolFromSmarts('nncn')]
    #     #'[n;$(n1nnnn1),$(n1nnncn1),$(n1nncn1),$(n1cnnn1)]') # '[n]1[n,n][n,n][n,n][n,n]1'
    # matches_n_poor_arom = []

    # for pattern_n_poor_arom in patterns_n_poor_arom:
    #     found_n_poor_arom, m_tmp = match_pattern(mol,pattern_n_poor_arom)
    #     if len(m_tmp) > 0:
    #         for m in m_tmp:
    #             matches_n_poor_arom.append(m)
    # Except everything that couldn't be neutralized
    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()]) # type: ignore
    for at_idx, q in enumerate(q0s):
        atom = mol.GetAtomWithIdx(at_idx) 
        map_idx = atom.GetAtomMapNum()
        # if q != 0.:
            # logger.info(f'Input molecule is charged at idx {at_idx}, map_idx {map_idx}!')
            # if map_idx not in except_indices:
                # except_indices.append(map_idx)

        if (atom.GetSymbol() == 'N') and (q == 0.) and not (atom.GetIsAromatic()):
            for match in matches_NphenNOO:
                if atom.GetIdx() in match:
                    if map_idx not in except_indices:
                        except_indices.append(map_idx)
                        NphenNOO_indices.append(map_idx)
        # if (atom.GetSymbol() == 'N') and (q == 0.) and (atom.GetIsAromatic()):
        #     for match in matches_n_poor_arom:
        #         if atom.GetIdx() in match:
        #             if map_idx not in except_indices:
        #                 except_indices.append(map_idx)
        #                 n_poor_arom_indices.append(map_idx)

    phosphate_found, phosphate_groups = has_phosphate(mol) # returns map indices

    if phosphate_found:
        logger.debug(f'phosphate ids: {phosphate_groups}')
        for p_idx, oh_ids in phosphate_groups.items():
            oh_ids = sorted(oh_ids)
            for map_idx in oh_ids:
                if map_idx not in except_indices:
                    except_indices.append(map_idx)

    invalid_amine_map_idx = has_invalid_amine(mol) # too short amine, breaks in molgpka

    if invalid_amine_map_idx > 0:
        if invalid_amine_map_idx not in except_indices:
            except_indices.append(invalid_amine_map_idx)

    return except_indices, phosphate_groups, invalid_amine_map_idx, NphenNOO_indices#, n_poor_arom_indices

def has_phosphate(mol: Mol) -> tuple[bool, dict[int, list[int]]]:
    """
    Detect phosphate groups and their protonable hydroxyl atoms.

    Searches the molecule for phosphate motifs and returns the atom map
    indices of the central phosphorus atoms along with the indices of
    attached protonable oxygen atoms.

    Returns
    -------
    found
        True if at least one phosphate group is detected.
    phosphate_groups
        Mapping from phosphate P atom map indices to protonable OH
        atom map indices.
    """

    pattern = Chem.MolFromSmarts("P(=O)(O)(O)")

    found, matches = match_pattern(mol,pattern)

    phosphate_groups: dict[int, list[int]] = {}

    for match in matches:
        # Find central P of phosphate
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "P":
                p_map_idx = atom.GetAtomMapNum()
                if p_map_idx not in phosphate_groups:
                    phosphate_groups[p_map_idx] = []
        # Find protonable O of phosphate
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
                oh_map_idx = atom.GetAtomMapNum()
                if oh_map_idx not in phosphate_groups[p_map_idx]:
                    phosphate_groups[p_map_idx].append(oh_map_idx)

    return found, phosphate_groups

def split_exceptions(
        indices: list[int],
        q_options: NDArray[np.int64],
        except_indices: list[int],
) -> tuple[list[int], NDArray[np.int64]]:
    """
    Remove exception indices from candidate protonation sites.

    Filters the list of candidate indices and their corresponding
    protonation options by removing sites that are handled separately
    (e.g., special functional groups).

    Returns
    -------
    indices_curated
        Filtered list of atom map indices.
    q_options_curated
        Protonation option matrix corresponding to the curated indices.
    """

    indices_curated = []
    q_options_curated = []
    for map_idx, q_option in zip(indices, q_options):
        if map_idx not in except_indices:
            indices_curated.append(map_idx)
            q_options_curated.append(q_option)
    q_options_curated_arr = np.array(q_options_curated)
    return indices_curated, q_options_curated_arr

def calc_phosphate_clusters(
    phosphate_groups: dict[int, list[int]],
    pH: float,
    matrix_def: str,
) -> tuple[list[list[str]], list[list[float]], list[list[int]]]:
    """
    Compute protonation state distributions for phosphate groups.

    Phosphates are treated as independent clusters with predefined
    pKa values. This function enumerates possible protonation states
    of the phosphate OH groups and calculates their equilibrium
    populations at the given pH.

    Returns
    -------
    state_strs_poh
        Protonation state encodings for each phosphate cluster.
    state_freqs_poh
        Corresponding microstate frequencies.
    oh_ids_poh
        Atom map indices of protonable OH atoms for each cluster.
    """

    state_strs_poh = []
    state_freqs_poh = []
    oh_ids_poh = []
    
    pka1 = 2.0
    pka2 = 6.5

    poh_acid_pkas_single: dict[str, list[float]] = {
        '0' : [pka1],
        '1' : [pka1],
    }

    base_lib_poh_single: dict[str, Any] = {
        '0': {},
        '1': {},
    }

    poh_acid_pkas_double: dict[str, list[float]] = {
        '00' : [pka2, pka2],
        '01' : [pka1, pka2],
        '10' : [pka2, pka1],
        '11' : [pka1, pka1],
    }

    base_lib_poh_double: dict[str, Any] = {
        '00': {},
        '01': {},
        '10': {},
        '11': {},
    }

    for p_idx, oh_ids in phosphate_groups.items():
        if len(oh_ids) == 1:
            state_strs = ['0','1']
            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            poh_acid_pkas = poh_acid_pkas_single
            base_lib_poh = base_lib_poh_single
        elif len(oh_ids) == 2:
            state_strs = ['00','01','10','11']
            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            poh_acid_pkas = poh_acid_pkas_double
            base_lib_poh = base_lib_poh_double
        else:
            logger.debug(f'Did not find protonable O for phosphate {p_idx}')
            continue
        acid_lib_poh: dict[str, dict[int, float]] = {}
        for key, val in poh_acid_pkas.items():
            acid_lib_poh[key] = {}
            for jdx, oh_id in enumerate(oh_ids):
                acid_lib_poh[key][oh_id] = val[jdx]

        logger.debug(oh_ids)
        logger.debug(acid_lib_poh)
        
        ps_all = calc_state_diffs(state_strs, state_vecs, oh_ids, base_lib_poh, acid_lib_poh, 
                                    pH=pH,matrix_def=matrix_def)
        
        state_strs_curated, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def)

        state_strs_poh.append(state_strs_curated)
        state_freqs_poh.append(state_freqs)
        oh_ids_poh.append(oh_ids)

    return state_strs_poh, state_freqs_poh, oh_ids_poh


# def is_methyl(atom):
#     # carbon with 3 hydrogens and only bonded to N
#     return (
#         atom.GetAtomicNum() == 6 and
#         atom.GetDegree() == 1 and
#         atom.GetTotalNumHs() == 3
#     )

# def is_ethyl(atom, parent_n):
#     # first carbon: CH2 attached to N
#     if atom.GetAtomicNum() != 6:
#         return False
    
#     if atom.GetTotalNumHs() != 2:
#         return False
    
#     neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() != parent_n.GetIdx()]
    
#     if len(neighbors) != 1:
#         return False
    
#     second = neighbors[0]
    
#     # second carbon must be CH3
#     return (
#         second.GetAtomicNum() == 6 and
#         second.GetTotalNumHs() == 3 and
#         second.GetDegree() == 1
#     )

# def has_invalid_amine(mol):
#     """ Special case amine with only ethyl or methyl (or no) substituents
#     e.g.
#     CCNCC
#     NCC
#     NC
#     CN(CC)CC
#     """
#     for atom in mol.GetAtoms():
#         if atom.GetAtomicNum() != 7:
#             continue
        
#         map_idx = atom.GetAtomMapNum()

#         neighbors = atom.GetNeighbors()
        
#         invalid = True
#         for nbr in neighbors:
#             if nbr.GetAtomicNum() == 1:
#                 continue
            
#             if is_methyl(nbr):
#                 continue
            
#             if is_ethyl(nbr, atom):
#                 continue
            
#             invalid = False
#             break
        
#         if invalid:
#             return map_idx
    
#     return 0



def short_alkyl(n_atom, mol):
    n_idx = n_atom.GetIdx()

    visited = set([n_idx])
    queue = deque([(n_atom, 0)])

    max_dist = 0

    while queue:
        atom, dist = queue.popleft()

        for nbr in atom.GetNeighbors():
            idx = nbr.GetIdx()

            if idx == n_idx:
                continue

            # distance tracking
            if idx not in visited:
                visited.add(idx)

                # atom type constraint
                if nbr.GetAtomicNum() not in (1, 6):
                    return False

                # ignore hydrogens in distance logic
                if nbr.GetAtomicNum() == 1:
                    continue

                new_dist = dist + 1
                max_dist = max(max_dist, new_dist)

                if max_dist > 2:
                    return False

                queue.append((nbr, new_dist))

    return True


def has_invalid_amine(mol):
    """ Special case amine with only ethyl or methyl or isopropyl (or no) substituents
    e.g.
    CCNCC
    NCC
    NC
    CN(CC)CC
    CC(C)NC(C)C
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue

        map_idx = atom.GetAtomMapNum()

        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                continue
            if nbr.GetAtomicNum() != 6:
                return 0

            if not short_alkyl(atom, mol):
                return 0
        return map_idx
    return 0

def calc_single_fixed_pka(
    pH: float,
    pka: float,
    ss_lower: int,
    matrix_def: str,
) -> tuple[list[str], list[float]]:
    """ Fix the pka for an amine with only methyl or ethyl substituents (molgpka bug)"""

    state_strs = [f'{ss_lower}',f'{ss_lower+1}']
    state_vecs = [unpack_vec(state_str) for state_str in state_strs]

    # pka = 10.4 # rough average value from different short amines in IUPAC list

    base_lib: dict[str, dict[int, float]] = {
        f'{ss_lower}' : {0: pka},
        f'{ss_lower+1}' : {0: pka},
    }

    acid_lib: dict[str, dict[int, float]] = {
        f'{ss_lower}' : {},
        f'{ss_lower+1}' : {},
    }

    ids = [0] # pseudo index

    ps_all = calc_state_diffs(state_strs, state_vecs, ids, base_lib, acid_lib, 
                                    pH=pH,matrix_def=matrix_def)

    state_strs_curated, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def)

    return state_strs_curated, state_freqs


