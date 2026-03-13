from .external.pka import predict_acid_base, load_model
from .transitions import *
from .postprocess import *
from .utils import *
from .coupling import *
from .special_cases import *

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import RegistrationHash

import numpy as np

import copy
import itertools
import os

from dataclasses import dataclass

from importlib import resources
pkg_base = resources.files('autoprot')

ROOT = f'{pkg_base}/data'

def preprocess(smiles_raw: str, verbose=False):
    """ 
    Build and preprocess mol.
    Neutralize all molecule charges. If not possible (e.g. quaternary amine),
    add to exclusion list to avoid (de-)protonation attempts. 
    """

    if verbose:
        print('Raw:')
        print(smiles_raw)
    mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    
    if verbose:
        print('Canonical')
        print(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if verbose:
        print('Formal charges before cleanup')
        charges = [at.GetFormalCharge() for at in mol.GetAtoms()]
        print(charges)

    mol = rdMolStandardize.Cleanup(mol)
    uncharger = rdMolStandardize.Uncharger(force=True)

    # load/save cycles to clean up the mol atom ordering
    mol = uncharger.uncharge(mol)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    if verbose:
        print('Formal charges after cleanup')
        symbols = [at.GetFormalCharge() for at in mol.GetAtoms()]
        print(symbols)

    return mol, smiles

def find_candidate_sites(base,acid,exclude_base_indices,exclude_acid_indices,pH,pH_band=8.,
                         verbose=False):
    """ Find possible (de-)protonation sites.
    Indices in the acid or base exclusion lists are removed from the
     respective q_options, but are kept in indices. """

    prot_candidates = list(base.keys()) # should be map idx
    deprot_candidates = list(acid.keys())

    indices = list(sorted(set(prot_candidates + deprot_candidates)))
    if verbose:
        print(f'relevant indices: {indices}')

    q_options = np.zeros((len(indices),3)) # deprot=0, stay=1, prot=2
    for rel_idx, map_idx in enumerate(indices):
        q_options[rel_idx,1] = 1 # always allow stay
        if map_idx in prot_candidates:
            if map_idx not in exclude_base_indices:
                if base[map_idx] >= pH - pH_band:
                    q_options[rel_idx,2] = 1 # allow protonation
        if map_idx in deprot_candidates:
            if map_idx not in exclude_acid_indices:
                if acid[map_idx] <= pH + pH_band:
                    q_options[rel_idx,0] = 1 # allow deprotonation
    return indices, q_options

def construct_state_vectors(q_options, cutoff_states):
    """ Enumerate all combinations of state_vectors allowed by q_options """

    q_options_nonzero = []
    for rel_idx, qs in enumerate(q_options):
        q_col = []
        for q_idx, q in enumerate(qs):
            if q == 1.:
                q_col.append(q_idx)
        if len(q_col) > 0.:
            q_options_nonzero.append(q_col)
    
    N_trial_vecs = np.prod([len(qs) for qs in q_options_nonzero])
    if N_trial_vecs > cutoff_states:
        return []
    else:
        state_vecs = np.array(list(itertools.product(*q_options_nonzero)))
        return state_vecs

#########################################
# rdkit mol object construction

def construct_mol(mol0, indices, state_vec):
    """ Make single rdkit mol object from 'neutral' mol0 and state vector """

    mol_cand = copy.deepcopy(mol0)

    smiles_cand = Chem.MolToSmiles(mol_cand, canonical=False)

    qs = state_vec - 1
    
    rw = Chem.RWMol(Chem.AddHs(mol_cand))

    for map_idx, q in zip(indices,qs):
        atom = get_atom_with_map_idx(rw, map_idx)
        atom.SetFormalCharge(int(q))
        if q == -1:
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    rw.RemoveAtom(nbr.GetIdx())
                    break

    mol_cand = Chem.RemoveHs(rw)
    Chem.SanitizeMol(mol_cand)
    smiles_cand = Chem.MolToSmiles(mol_cand, canonical=False)

    return mol_cand, smiles_cand


#############################################################################################
# Cluster tests and operations



def combine_clusters(state_strs_clusters, state_freqs_clusters, indices_clusters, 
                     sfreq_cutoff_individual=0.01,sfreq_cutoff_combined=0.001,verbose=False):
    """ 
    Combine state frequency results from independent pKa clusters in molecule. 
    The combined probability for microstate A in cluster 1 and B in cluster 2 is:
    p(AB) = p(A)p(B) 
    """

    # Cull the state_strs per cluster a bit before combining.
    # This is quite conservative (everything with at least 1% freq in that cluster)

    cluster_state_ids = []

    if verbose:
        print(state_strs_clusters)
        print(state_freqs_clusters)

    for state_freqs in state_freqs_clusters:
        cluster_state_ids.append([])
        for s_idx, s_freq in enumerate(state_freqs):
            if s_freq >= sfreq_cutoff_individual:
                cluster_state_ids[-1].append(s_idx)

    combinations = list(itertools.product(*cluster_state_ids))
    if verbose:
        print(f'N microstate combinations from clusters: {len(combinations)}')

    state_strs = []
    state_freqs = []
    indices = []
    for indices_cluster in indices_clusters:
        indices.extend(indices_cluster) # This requires non-overlapping clusters!

    ps = np.argsort(indices)
    indices = [indices[p] for p in ps]
    
    for s_idxs in combinations:
        state_str = ''
        state_freq = 1.
        for c_idx, s_idx in enumerate(s_idxs):
            state_str += state_strs_clusters[c_idx][s_idx]
            state_freq *= state_freqs_clusters[c_idx][s_idx]
        state_str = sort_string(state_str,ps) # match sorted indices                 
        if state_freq >= sfreq_cutoff_combined:
            state_strs.append(state_str)
            state_freqs.append(state_freq)

    if verbose:
        print(f'N chosen microstate combinations: {len(state_strs)}')
    # Correct freqs for removal of very unlikely states
    state_freqs = np.array(state_freqs)
    state_freqs /= np.sum(state_freqs)

    state_freqs_lib = {}

    for state_str, state_freq in zip(state_strs, state_freqs):
        state_freqs_lib[state_str] = state_freq

    return indices, state_strs, state_freqs_lib

def smiles2hash(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    return RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(Chem.MolFromSmiles(smiles)))

def mol2hash(mol):
    return RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(mol))

def calc_hashes(state_strs,mols_lib):
    hashes = []
    for state_str in state_strs:
        mol = mols_lib[state_str]
        hash = mol2hash(mol)
        hashes.append(hash)
    return hashes

def calc_symmetry(state_strs, state_freqs_lib, mols_lib,verbose=False):
    state_hashes = calc_hashes(state_strs,mols_lib)
    state_dict = {}

    for state_str, state_hash in zip(state_strs,state_hashes):
        if state_hash in state_dict:
            state_dict[state_hash].append(state_str)
        else:
            state_dict[state_hash] = [state_str]

    if verbose:
        print(state_dict)

    state_strs_symm = []
    state_freqs_symm = []

    for state_hash, state_strs_per_hash in state_dict.items():
        state_strs_sorted = sorted(state_strs_per_hash)
        state_strs_symm.append(state_strs_sorted[0])
        state_freq = 0.
        for state_str in state_strs_per_hash:
            state_freq += state_freqs_lib[state_str]
        state_freqs_symm.append(state_freq)
    
    state_strs = state_strs_symm
    state_freqs = state_freqs_symm

    return state_strs, state_freqs



def calc_macro_props(state_strs, state_freqs, mols_lib):
    """ Net charge as weighted sum over microstate charges """

    state_qs = {}
    freqs_macro = {}
    net_charge = 0.       
    for state_str, state_freq in zip(state_strs, state_freqs):
        state_q = Chem.GetFormalCharge(mols_lib[state_str])
        state_qs[state_str] = state_q
        if state_q in freqs_macro:
            freqs_macro[state_q] += state_freq
        else:
            freqs_macro[state_q] = state_freq
        net_charge += state_q * state_freq
    return net_charge, state_qs, freqs_macro

###########


def combine_pkas_macro(pHs, freqs_macro_all):
    
    pkas_macro = {}
    pkas_weights = {}

    for pH, freqs_macro in zip(pHs,freqs_macro_all):
        qs_sorted = sorted(freqs_macro.keys())
        for q in qs_sorted:
            if q+1 in qs_sorted:
                freq1 = freqs_macro[q]
                freq2 = freqs_macro[q+1]
                pka_macro = np.log10(freq2/freq1) + pH
                pka_weight = 1./(freq1**2 + freq2**2)
                if q in pkas_macro:
                    pkas_macro[q].append(pka_macro)
                    pkas_weights[q].append(pka_weight)
                else:
                    pkas_macro[q] = [pka_macro]
                    pkas_weights[q] = [pka_weight]

    pkas_combined = {}

    for q, pkas in pkas_macro.items():
        ws = pkas_weights[q]
        pka_comb = float(np.average(pkas,weights=ws))
        pkas_combined[q] = pka_comb
    return pkas_combined

###########

@dataclass
class ParamsOutput:
    name: str
    path_out : str
    path_figs: str
    fout_csv: str
    append: bool
    notebook: bool

class Autoprot:
    def __init__(self, smiles_raw, **kwargs):
        self.smiles_raw = smiles_raw

        defaults = {
            'name': 'molecule',
            'cutoff_states': 4000,
            'device': 'cpu',
            'pH_output': 7.,
            'pH_band': 8.,
            'pHs': np.arange(0,14.1,0.5),
            'fout_csv': f'results.csv',
            'append': False,
            'notebook': False,
            'sfreq_cutoff_individual': 0.01,
            'sfreq_cutoff_combined': 0.001,
            'matrix_def': 'dG',
            'export_opti_sdf': False,
            'path_out': 'output',
            'path_figs': 'figures',
            'cutoff_export': 0.5,
            'verbose': False
        }

        args_output = [
            'name','path_out','path_figs','fout_csv',
            'append','notebook']
        p = {}
        for arg in args_output:
            p[arg] = kwargs.get(arg,defaults[arg])
        self.params_out = ParamsOutput(**p)

        for key, val in defaults.items():
            if key not in args_output:
                setattr(self, key, kwargs.get(key, val))

    def run(self):
                # name,smiles_raw,pH_output=7,cutoff_states=4000,device='cpu',
                # pH_band=8.,pHs = np.arange(0,14.1,0.5),
                # path_out='output',path_figs='figures',
                # verbose=False,cutoff_export=0.5,
                # fout_csv='out.csv',append=True,notebook=False,
                # sfreq_cutoff_individual=0.01,
                # sfreq_cutoff_combined=0.001,
                # matrix_def='dG',export_opti_sdf=False):
        if self.verbose:
            print(self.params_out.name)
            print(self.smiles_raw, flush=True)

        os.makedirs(self.params_out.path_out, exist_ok=True)
        os.makedirs(self.params_out.path_figs, exist_ok=True)

        # molgpka ML models
        model_file_base = f'{ROOT}/weight_base.pth'
        model_file_acid = f'{ROOT}/weight_acid.pth'
        self.model_base = load_model(model_file_base,device=self.device)
        self.model_acid = load_model(model_file_acid,device=self.device)

        self.base_lib = {}
        self.acid_lib = {}
        self.smiles_lib = {}
        self.mols_lib = {}

        net_charges = []
        state_freqs_all = {}
        freqs_macro_all = []

        self.mol0, self.smiles0 = preprocess(self.smiles_raw, verbose=self.verbose)
        exclude_base_indices, exclude_acid_indices = add_exclusions(self.mol0, verbose=self.verbose)
        except_indices, phosphate_groups = add_exceptions(self.mol0, verbose=self.verbose)

        if self.verbose:
            print('Processed:')
            print(self.smiles0)
            print(f'Exclude base indices: {exclude_base_indices}')
            print(f'Exclude acid indices: {exclude_acid_indices}')
            print(f'Except indices: {except_indices}')

        if len(phosphate_groups) > 0:
            print(phosphate_groups)

        mol0_h = Chem.rdmolops.AddHs(self.mol0)

        base0, acid0 = predict_acid_base(mol0_h,
                                         self.model_base,self.model_acid,
                                         device=self.device,verbose=self.verbose) # returns pkas for map indices

        for pH_idx, pH in enumerate(self.pHs): #,total=len(pHs)):#,total=len(pHs)):
            if self.verbose:
                print('='*50)
                print(f'pH: {pH}',flush=True)
            self.indices0, q_options0 = find_candidate_sites(base0, acid0, exclude_base_indices, exclude_acid_indices,
                                                        pH, pH_band=self.pH_band, verbose=False)
            if self.verbose:
                print(f'indices0: {self.indices0}')
                print(f'q_options0: {q_options0}')
            self.indices0_str = pack_indices(self.indices0)

            indices0_curated, q_options0 = split_exceptions(self.indices0, q_options0, except_indices)

            if self.verbose:
                print(f'curated indices0: {indices0_curated}')
                print(f'curated q_options0: {q_options0}')

            # Screen coupling between residues
            clusters = self.screen_clusters(indices0_curated, q_options0)
            if self.verbose:
                print(f'Clusters: {clusters}')
            state_freqs_clusters = []
            state_strs_clusters = []
            indices_clusters = []
            
            for c_idx, cluster in enumerate(clusters):

                # atom indices for cluster
                indices = [indices0_curated[c] for c in cluster]
                indices_str = pack_indices(indices)

                q_options = q_options0[cluster]
                state_vecs = construct_state_vectors(q_options, self.cutoff_states)

                state_strs = calc_state_strs(state_vecs)

                self.construct_mols(state_strs, state_vecs, indices)
                self.run_acid_base_calcs(state_strs,state_vecs,indices)

                ps_all = calc_state_diffs(
                    state_strs, state_vecs, indices,
                    self.base_lib[indices_str], self.acid_lib[indices_str], 
                    pH=pH,matrix_def=self.matrix_def,verbose=self.verbose)
                
                state_strs, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,self.matrix_def)

                state_strs_clusters.append(state_strs)
                state_freqs_clusters.append(state_freqs)
                indices_clusters.append(indices)

            # Inject phosphate clusters:
            if len(phosphate_groups) > 0:
                state_strs_poh, state_freqs_poh, oh_ids_poh = calc_phosphate_clusters(phosphate_groups,pH,self.matrix_def,
                                                                verbose=self.verbose)
                for state_strs, state_freqs, oh_ids in zip(state_strs_poh,state_freqs_poh,oh_ids_poh):
                    state_strs_clusters.append(state_strs)
                    state_freqs_clusters.append(state_freqs)
                    indices_clusters.append(oh_ids)

            # Combine clusters and their frequencies
            indices, state_strs, state_freqs_lib = combine_clusters(
                state_strs_clusters, state_freqs_clusters, indices_clusters, 
                sfreq_cutoff_individual=self.sfreq_cutoff_individual,
                sfreq_cutoff_combined=self.sfreq_cutoff_combined,
                verbose=self.verbose)

            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            self.construct_mols(state_strs, state_vecs, indices)

            indices_str = pack_indices(indices)
            if indices_str != self.indices0_str:
                raise ValueError(f"indices_str {indices_str} not equal self.indices0_str {self.indices0_str}")
            # Symmetry
            state_strs, state_freqs = calc_symmetry(
                state_strs, state_freqs_lib, self.mols_lib[indices_str], verbose=self.verbose)

            # Macro-pka properties
            net_charge, state_qs, freqs_macro = calc_macro_props(
                state_strs, state_freqs, self.mols_lib[indices_str])
            net_charges.append(net_charge)
            freqs_macro_all.append(freqs_macro)

            # Add to results for pH scan
            for state_str, state_freq in zip(state_strs,state_freqs):
                if state_str not in state_freqs_all:
                    state_freqs_all[state_str] = np.zeros(len(self.pHs))
                state_freqs_all[state_str][pH_idx] = state_freq

            # Output pH-specific results for pH_output
            if pH == self.pH_output:
                self.make_pH_specific_output(state_strs, state_freqs, state_qs, indices)

        # Plotting of pH scan
        net_charges = np.round(np.array(net_charges),decimals=4)

        pkas_combined = combine_pkas_macro(self.pHs, freqs_macro_all)

        for idx, (q, pka) in enumerate(pkas_combined.items()):
            print(f'pKa{idx+1} | {q+1} --> {q} | {pka:.3f}')

        if len(pkas_combined) > 0:
            export_macro_pkas(pkas_combined,self.params_out)

        # reduce number of microstates for plotting
        N_relevant_states, state_strs_relevant, sfreqs_relevant, mols_relevant, sfreqs_not_relevant = calc_relevant_states(
                state_freqs_all, self.mols_lib[self.indices0_str], verbose=self.verbose)

        if N_relevant_states > 0:
            plot_pH_scan(
                    self.params_out.name, indices, state_strs_relevant, sfreqs_relevant, self.pHs, net_charges, 
                    sfreqs_not_relevant, pkas_combined, path=self.params_out.path_figs,verbose=self.verbose)
            plot_relevant_states(mols_relevant, self.params_out)
            compose_image(N_relevant_states, self.params_out)


    #########################

    def coupling_assay(self, indices, q_options, coupling_cutoff):
        """ Sensitivity analysis. (De-)protonate every site one by one and assess if pKa values of other sites change """

        indices_str = pack_indices(indices)

        state_vecs = construct_state_vectors_single(indices, q_options)
        state_strs = calc_state_strs(state_vecs)
        self.construct_mols(state_strs, state_vecs, indices)
        self.run_acid_base_calcs(state_strs, state_vecs, indices)

        state_str0 = state_strs[0]
        base_pka_diffs = {}
        acid_pka_diffs = {}
        for state_str1 in state_strs[1:]:
            base_pka_diffs[state_str1], acid_pka_diffs[state_str1] = compare_pkas(
                    indices, q_options, state_str0, state_str1, 
                    self.base_lib[indices_str], self.acid_lib[indices_str])
        
        coupling_matrix = construct_coupling_matrix(
                indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, 
                coupling_cutoff)
        clusters = cluster_coupling_matrix(coupling_matrix)
        return clusters

    def screen_clusters(self, indices0, q_options0):
        """ Screen fragmentation of molecule into different pKa clusters """
        accept_clusters = False
        coupling_cutoff = 0.0
        while not accept_clusters:
            if self.verbose:
                print(f'coupling cutoff: {coupling_cutoff}')
            accept_clusters = True
            clusters = self.coupling_assay(indices0, q_options0, coupling_cutoff)
            for c_idx, cluster in enumerate(clusters):
                q_options = q_options0[cluster]
                state_vecs = construct_state_vectors(q_options, self.cutoff_states)
                N_states = len(state_vecs)
                if N_states > self.cutoff_states: # This should never happen, as construct_state_vectors should return an empty list in that case.
                    raise
                if (N_states == 0):
                    accept_clusters = False
                    coupling_cutoff += 0.2
        if coupling_cutoff > 1.5:
            print(f'Coupling cutoff high: {coupling_cutoff}')
        return clusters

    def construct_mols(self, state_strs, state_vecs, indices):#, mols_lib, smiles_lib):
        """ Make rdkit mol objects for all protonation states defined in state_strs """

        indices_str = pack_indices(indices)
        if indices_str not in self.mols_lib:
            self.mols_lib[indices_str] = {}
            self.smiles_lib[indices_str] = {}

        for state_str, state_vec in zip(state_strs, state_vecs):
            if state_str not in self.mols_lib[indices_str]:
                mol_cand, smiles_cand = construct_mol(self.mol0, indices, state_vec)
                mol_cand.SetProp("_Name",state_str)
                self.mols_lib[indices_str][state_str] = mol_cand
                self.smiles_lib[indices_str][state_str] = smiles_cand
        # return mols_lib, smiles_lib

    ###################################
    # Acid-base calculation

    def run_acid_base_calcs(self,state_strs,state_vecs,indices):
        """ Add base and acid calculation results for all state_strs into base_lib and acid_lib """

        indices_str = pack_indices(indices)
        if indices_str not in self.base_lib:
            self.base_lib[indices_str] = {}
        if indices_str not in self.acid_lib:
            self.acid_lib[indices_str] = {}

        for state_str, state_vec in zip(state_strs,state_vecs):
            if state_str in self.base_lib[indices_str]:
                continue

            if self.verbose:
                print(state_str)

            state_vec_base = np.maximum(state_vec,1) # disregard de-protonations of other sites to assess base probability
            state_str_base = pack_vec(state_vec_base)

            mol_base = self.mols_lib[indices_str][state_str_base]
            mol_base_h = Chem.rdmolops.AddHs(mol_base)

            base_tmp, _ = predict_acid_base(
                    mol_base_h,self.model_base,self.model_acid,device=self.device,
                    pred_acid=False,verbose=self.verbose)
            base = {}
            for map_idx, b in base_tmp.items():
                if map_idx not in indices:
                    continue
                rel_idx = indices.index(map_idx)
                if state_vec[rel_idx] == 1: # Only consider predicted protonation/de-protonation predictions from neutral state
                    base[map_idx] = b

            state_vec_acid = np.minimum(state_vec,1) # disregard protonations of other sites to assess acid probability
            state_str_acid = pack_vec(state_vec_acid)

            mol_acid = self.mols_lib[indices_str][state_str_acid]
            mol_acid_h = Chem.rdmolops.AddHs(mol_acid)

            _, acid_tmp = predict_acid_base(
                    mol_acid_h,self.model_base,self.model_acid,device=self.device,
                    pred_base=False,verbose=self.verbose)
            
            acid = {}
            for map_idx, a in acid_tmp.items():
                if map_idx not in indices:
                    continue
                rel_idx = indices.index(map_idx)
                if state_vec[rel_idx] == 1: # Only consider predicted protonation/de-protonation predictions from neutral state
                    acid[map_idx] = a

            self.base_lib[indices_str][state_str] = base
            self.acid_lib[indices_str][state_str] = acid
        
    def make_pH_specific_output(self, state_strs, state_freqs, state_qs, indices):
                # state_strs, state_freqs, cutoff_export,pH,mols_lib,smiles_lib,
                # name, state_qs, path_out, fout_csv, append, export_opti_sdf, path_figs,
                # verbose=False):
        """ Plot properties of microstates for pH_output """

        # Max freq
        idx_max = np.argmax(state_freqs)
        state_freq_max = np.max(state_freqs)
        state_str_opti = state_strs[idx_max]

        # Select states for pH-specific export
        state_strs_export = []
        state_freqs_export = []
        for state_str, state_freq in zip(state_strs, state_freqs):
            if state_freq > self.cutoff_export * state_freq_max: # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export.append(state_freq)

        state_freqs_export = np.array(state_freqs_export)
        ps = np.argsort(state_freqs_export)[::-1] # Sort by highest probability

        state_freqs_export = state_freqs_export[ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        indices_str = pack_indices(indices)
        self.check_chiral_consistency(state_strs, indices)
            # state_strs_export, mols_lib[indices_str], smiles_lib[indices_str])

        if self.verbose:
            print(f'Export at pH {self.pH_output}:',flush=True)
            for e_idx, (state_str, sfreq) in enumerate(zip(state_strs_export, state_freqs_export)):
                print(e_idx, state_str, sfreq)
        export_csv(state_strs_export,self.smiles_lib[indices_str],state_freqs_export,state_qs,self.params_out)
        if self.export_opti_sdf:
            export_sdf(state_strs_export,self.mols_lib[indices_str],self.params_out)
            plot_optimal_state(self.mols_lib[indices_str][state_strs_export[0]],self.params_out)
        if self.verbose:
            print(f'Optimal smiles for pH {self.pH_output}: {self.smiles_lib[indices_str][state_str_opti]}')

    def check_chiral_consistency(self, state_strs, indices):
        indices_str = pack_indices(indices)
        for state_str in state_strs:
            mol = self.mols_lib[indices_str][state_str]

            mol_h = Chem.AddHs(mol)

            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)
            if cid != 0:
                print(f'WARNING: Need to remove chirality for embedding for {state_str}!')
                for atom in mol_h.GetAtoms():
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)

            if cid != 0:
                raise ValueError(f'{state_str} could not be embedded.')
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            
            smiles = Chem.MolToSmiles(mol)
            # print(smiles)
            self.mols_lib[indices_str][state_str] = mol
            self.smiles_lib[indices_str][state_str] = smiles
