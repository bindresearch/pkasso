from __future__ import division, unicode_literals

import os
from importlib import resources

import pandas as pd
from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import Mol

pkg_base = resources.files('pkasso')
root = f'{pkg_base}/data'

SMARTS_FILE = os.path.join(root, "smarts_pattern.tsv")

def split_acid_base_pattern() -> tuple[DataFrame, DataFrame]:
    """
    Load SMARTS patterns and split them into acid and base DataFrames.
    """
    df_smarts = pd.read_csv(SMARTS_FILE, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base

def unique_acid_match(matches: list[list[int]]) -> list[list[int]]:
    """
    Remove duplicate single-atom matches and combine with multi-atom matches.
    """
    single_matches = list(set([m[0] for m in matches if len(m)==1]))
    double_matches = [m for m in matches if len(m)==2]
    single_matches_l = [[j] for j in single_matches]
    double_matches.extend(single_matches_l)
    return double_matches

def match_acid(df_smarts_acid: DataFrame, mol: Mol) -> list[int]:
    """
    Find acid pattern matches in a molecule and return matched atom indices.
    """
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_acid.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def match_base(df_smarts_base: DataFrame, mol: Mol) -> list[int]:
    """
    Find base pattern matches in a molecule and return matched atom indices.
    """
    matches = []
    for idx, name, smarts, indexs, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        index_split = indexs.split(",")
        for index in index_split:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def get_ionization_aid(
    mol: Mol,
    acid_or_base: str,
) -> list[int]:
    """
    Identify ionization-relevant atom indices in a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Input molecule.
    acid_or_base : {"acid", "base", None}, optional
        If "acid", return only acid matches.
        Otherwise return base matches.

    Returns
    -------
    List[int]
        Matched atom indices.
    """
    df_smarts_acid, df_smarts_base = split_acid_base_pattern()

    if not mol:
        raise RuntimeError("No mol found for get_ionization_aid")
    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)
    if acid_or_base == "acid":
        return acid_matches
    elif acid_or_base == 'base':
        return base_matches
    else:
        raise ValueError