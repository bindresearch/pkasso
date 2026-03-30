from .external.pka import predict_acid_base, load_model
from .transitions import *
from .postprocess import *
from .utils import *
from .coupling import *
from .special_cases import *

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import RegistrationHash
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger
from rdkit.Chem.Draw import MolToFile, MolsToGridImage

RDLogger.DisableLog("rdApp.*")

import numpy as np
from numpy.typing import NDArray

import copy
import itertools
import os

from dataclasses import dataclass

from importlib import resources
pkg_base = resources.files('autoprot')

ROOT = f'{pkg_base}/data'

def preprocess(smiles_raw: str, verbose: bool = False) -> tuple[Mol,  str]:
    """ 
    Construct and standardize an RDKit molecule from a SMILES string.
    Charges that cannot be neutralized (e.g., quaternary ammonium) are preserved.
    Atom map numbers are assigned to preserve mapping when the molecule gets changed
    (re-ordered, protonated, de-protonated).

    Parameters
    ----------
    smiles_raw : str
        Input SMILES string representing the molecule.
    verbose : bool, optional
        Prints verbose information. Default is False.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        The standardized RDKit molecule with atom map numbers set to
        1-based indices.
    smiles : str
        Canonical SMILES representation of the processed molecule.
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
        charges = [at.GetFormalCharge() for at in mol.GetAtoms()] # type: ignore
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

    for atom in mol.GetAtoms(): # type: ignore
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    if verbose:
        print('Formal charges after cleanup')
        symbols = [at.GetFormalCharge() for at in mol.GetAtoms()] # type: ignore
        print(symbols)

    return mol, smiles

def find_candidate_sites(
        base: dict[int, float],
        acid: dict[int, float],
        exclude_base_indices: list[int],
        exclude_acid_indices: list[int],
        pH: float,
        pH_band: float = 8.,
        verbose: bool = False,
) -> tuple[list[int], NDArray[np.int64]]:
    """
    Determine possible protonation and deprotonation sites for a molecule.
    Candidate atom indices are derived from predicted basic and acidic sites.

    Parameters
    ----------
    base : dict
        Mapping of atom map indices to predicted basic pKa values.
    acid : dict
        Mapping of atom map indices to predicted acidic pKa values.
    exclude_base_indices : list[int]
        Atom indices that must not be considered for protonation.
    exclude_acid_indices : list[int]
        Atom indices that must not be considered for deprotonation.
    pH : float
        Target pH value used to evaluate protonation states.
    pH_band : float, optional
        Allowed pKa tolerance around the pH when determining candidate
        sites. Default is 8.
    verbose : bool, optional
        If True, prints the list of relevant atom indices.

    Returns
    -------
    indices : list[int]
        Sorted atom map indices considered for protonation state changes.
    q_options : NDArray[np.int64]
        Array of shape (n_sites, 3) indicating allowed states per site:
        [deprotonated, unchanged, protonated].
    """

    prot_candidates = list(base.keys()) # should be map idx
    deprot_candidates = list(acid.keys())

    indices = list(sorted(set(prot_candidates + deprot_candidates)))
    if verbose:
        print(f'relevant indices: {indices}')

    q_options = np.zeros((len(indices),3),dtype=np.int64) # deprot=0, stay=1, prot=2
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

def construct_state_vectors(
    q_options: NDArray[np.int64],
    cutoff_states: int,
) -> list[NDArray[np.int64]]:
    """
    Enumerate all valid protonation state vectors given allowed site options.

    For each site, the allowed states are extracted from ``q_options`` and all
    combinations are generated via a Cartesian product. If the total number of
    possible combinations exceeds ``cutoff_states``, enumeration is skipped and
    an empty list is returned.

    Parameters
    ----------
    q_options :  NDArray[np.int64]
        Array of shape (n_sites, 3) indicating allowed states per site,
        where columns correspond to [deprotonated, unchanged, protonated]
        and entries are 1 (allowed) or 0 (disallowed).
    cutoff_states : int
        Maximum number of state combinations to enumerate.

    Returns
    -------
    list[NDArray[np.int64]]
        Array of shape (n_states, n_sites) containing all valid state vectors,
        or an empty list if the number of combinations exceeds ``cutoff_states``.
    """

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
        return []#np.array([])
    else:
        state_vecs = list([np.array(x) for x in list(itertools.product(*q_options_nonzero))])
        return state_vecs

#########################################
# rdkit mol object construction

def construct_mol(mol0: Mol, indices: list[int], state_vec: NDArray[np.int64]) -> tuple[Mol, str]:
    """
    Construct a protonation-state-specific molecule from a reference molecule.

    The function applies the protonation/deprotonation state encoded in
    ``state_vec`` to the atoms specified by ``indices`` (atom map numbers).
    Formal charges are adjusted accordingly and hydrogens are added or removed
    where required. The resulting molecule is sanitized and returned together
    with a SMILES representation.

    Parameters
    ----------
    mol0 : rdkit.Chem.rdchem.Mol
        Reference molecule (neutral standardized structure)
        with atom map numbers assigned.
    indices : list[int]
        Atom map indices corresponding to the sites whose states are
        defined in ``state_vec``.
    state_vec :  NDArray[np.int64]
        Protonation state vector for the selected sites. Values are encoded
        as [0, 1, 2] corresponding to [deprotonated, unchanged, protonated].

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule with the specified protonation states applied.
    smiles : str
        Non-canonical SMILES representation of the constructed molecule.
    """

    mol_cand = copy.deepcopy(mol0)

    smiles_cand = Chem.MolToSmiles(mol_cand, canonical=False)

    qs = state_vec - 1
    
    rw = Chem.RWMol(Chem.AddHs(mol_cand))

    for map_idx, q in zip(indices,qs):
        atom = get_atom_with_map_idx(rw, map_idx)
        if atom is None:
            raise
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

def combine_clusters(
    state_strs_clusters: list[list[str]],
    state_freqs_clusters: list[NDArray[np.float64]],
    indices_clusters: list[list[int]],
    sfreq_cutoff_individual: float = 0.01,
    sfreq_cutoff_combined: float = 0.001,
    verbose: bool = False
) -> tuple[list[int], list[str], dict[str, float]]:

    """
    Combine microstate probabilities from independent pKa clusters.

    Microstates from different clusters are combined assuming statistical
    independence, with the combined microstate probability calculated as:

        p(AB) = p(A) * p(B)

    Prior to combination, low-frequency states are filtered within each
    cluster using ``sfreq_cutoff_individual``. After generating all possible
    combinations, a second filter is applied using
    ``sfreq_cutoff_combined``. Remaining probabilities are renormalized to
    sum to 1.

    Parameters
    ----------
    state_strs_clusters : list[str]
        Microstate string labels for each cluster.
    state_freqs_clusters : list[NDArray[np.float64]]
        Corresponding state frequency arrays for each cluster.
    indices_clusters : list[list[int]]
        Atom indices associated with each cluster (must be non-overlapping).
    sfreq_cutoff_individual : float, optional
        Minimum frequency required for a state to be considered during
        cluster-wise filtering. Default is 0.01.
    sfreq_cutoff_combined : float, optional
        Minimum frequency required for a combined microstate to be kept.
        Default is 0.001.
    verbose : bool, optional
        If True, prints intermediate information during processing.

    Returns
    -------
    indices : list[int]
        Sorted list of all atom indices across clusters.
    state_strs : list[str]
        Combined microstate string representations passing frequency filters.
    state_freqs_lib : dict[str, float]
        Dictionary mapping microstate strings to their normalized
        probabilities. Used downstream for calc_symmetry().
    """

    # Cull the state_strs per cluster a bit before combining.
    # This is quite conservative (everything with at least 1% freq in that cluster)

    cluster_state_ids: list[list[int]] = []

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
    state_freqs_list = []
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
            state_freqs_list.append(state_freq)

    if verbose:
        print(f'N chosen microstate combinations: {len(state_strs)}')
    # Correct freqs for removal of very unlikely states
    state_freqs = np.array(state_freqs_list)
    state_freqs /= np.sum(state_freqs)

    state_freqs_lib = {}

    for state_str, state_freq in zip(state_strs, state_freqs):
        state_freqs_lib[state_str] = state_freq

    return indices, state_strs, state_freqs_lib

def smiles2hash(smiles: str | None) -> str | None:
    """
    Generate a registration hash from a SMILES string.

    Returns None if the input is None.
    """

    if smiles is None:
        return None
    return RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(Chem.MolFromSmiles(smiles)))

def mol2hash(mol: Mol) -> str:
    """
    Generate a registration hash from an RDKit molecule.
    """

    return RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(mol))

def calc_hashes(state_strs: list[str], mols_lib: dict[str, Mol]) -> list[str]:
    """
    Compute registration hashes for a set of microstates.

    Parameters
    ----------
    state_strs : iterable of str
        Microstate identifiers used as keys in ``mols_lib``.
    mols_lib : dict[str, rdkit.Chem.rdchem.Mol]
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    list[str]
        Registration hashes corresponding to the input microstates,
        in the same order as ``state_strs``.
    """

    hashes = []
    for state_str in state_strs:
        mol = mols_lib[state_str]
        hash = mol2hash(mol)
        hashes.append(hash)
    return hashes

def calc_symmetry(
    state_strs: list[str],
    state_freqs_lib: dict[str, float],
    mols_lib: dict[str, Mol],
    verbose: bool = False,
) -> tuple[list[str], list[float]]:
    """
    Merge symmetry-equivalent microstates based on molecular hashes.

    Microstates that share the same registration hash are grouped together.
    Their probabilities are summed, and a single representative state
    (alphabetically sorted) is retained per group.

    Parameters
    ----------
    state_strs : list[str]
        Microstate identifiers.
    state_freqs_lib : dict[str, float]
        Mapping from microstate strings to their probabilities.
    mols_lib : dict[str, Mol]
        Mapping from microstate strings to RDKit molecule objects.
    verbose : bool, optional
        If True, prints intermediate grouping information.

    Returns
    -------
    tuple[list[str], list[float]]
        Symmetry-reduced microstate strings and their corresponding
        combined frequencies.
    """

    state_hashes = calc_hashes(state_strs,mols_lib)
    state_dict: dict[str, list[str]] = {}

    for state_str, state_hash in zip(state_strs,state_hashes):
        if state_hash in state_dict:
            state_dict[state_hash].append(state_str)
        else:
            state_dict[state_hash] = [state_str]

    if verbose:
        print(state_dict)

    state_strs_symm: list[str] = []
    state_freqs_symm: list[float] = []

    for state_hash, state_strs_per_hash in state_dict.items():
        state_strs_sorted = sorted(state_strs_per_hash)
        state_strs_symm.append(state_strs_sorted[0])
        state_freq = 0.
        for state_str in state_strs_per_hash:
            state_freq += state_freqs_lib[state_str]
        state_freqs_symm.append(state_freq)

    return state_strs_symm, state_freqs_symm

def calc_macro_props(
    state_strs: list[str],
    state_freqs: list[float],
    mols_lib: dict[str, Mol],
) -> tuple[float, dict[str, int], dict[int, float]]:
    """
    Compute macrostate properties from weighted microstate contributions.

    The net charge is calculated as the frequency-weighted sum of
    microstate formal charges. Additionally, per-microstate charges
    and the aggregated charge distribution are returned.

    Parameters
    ----------
    state_strs : list[str]
        Microstate identifiers.
    state_freqs : list[float]
        Corresponding normalized microstate probabilities.
    mols_lib : dict[str, Mol]
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    tuple[float, dict[str, int], dict[int, float]]
        - Net charge (frequency-weighted sum)
        - Dictionary mapping microstate strings to formal charges
        - Dictionary mapping formal charges to their total frequency
    """

    state_qs: dict[str, int] = {}
    freqs_macro: dict[int, float] = {}
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

def combine_pkas_macro(pHs: NDArray[np.float64], freqs_macro_all: list[dict[int, float]]) -> dict[int, float]:
    """
    Estimate macrostate pKa values from charge-resolved frequency data.

    For each pH value, adjacent charge states (q and q+1) are used to
    compute a pKa estimate:

        pKa = log10(freq(q+1) / freq(q)) + pH

    Multiple pKa estimates for the same charge transition are combined
    using weighted averaging. pKa estimates from evaluations with pH values
    close to the pKa are weighted higher.

    Parameters
    ----------
    pHs : NDArray[np.float64]
        Array of pH values corresponding to the frequency datasets.
    freqs_macro_all : list[dict[int, float]]
        List of macrostate frequency dictionaries, one per pH value.
        Each dictionary maps formal charge (int) to its frequency.

    Returns
    -------
    dict[int, float]
        Combined macrostate pKa values indexed by charge state.
        Represents pKa between q and q+1.
    """

    pkas_macro: dict[int, list[float]] = {}
    pkas_weights: dict[int, list[float]] = {}

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

    pkas_combined: dict[int, float] = {}

    for q, pkas in pkas_macro.items():
        ws = pkas_weights[q]
        pka_comb = float(np.average(pkas,weights=ws))
        pkas_combined[q] = pka_comb
    return pkas_combined

###########

@dataclass
class Autoprot:
    """
    Autoprot pipeline for protonation state prediction.

    Parameters
    ----------
    smiles_raw : str
        Input SMILES string of the molecule to be processed.
    **kwargs : object
        Optional configuration parameters. Supported keys include:

        If mode == 'single': Run a microstate frequency prediction 
        at the given pH value.
        If mode == 'scan': Like 'single', but in addition plot frequencies 
        from a pH scan.

        Pipeline parameters:
            name, cutoff_states, device, pH, pH_band,
            pHs, sfreq_cutoff_individual, sfreq_cutoff_combined,
            matrix_def, no_sdf, outfolder,
            cutoff_export, verbose

        Output-related parameters:
            name, outfolder, outfolder, fout_csv, append

    """
    smiles: str

    name: str = 'molecule'
    # outfolder: str = 'output'

    save: bool = False
    path: str = '.'

    # fout_csv: str = 'autoprot_results.csv'
    # append_csv: bool = False

    # batch: str | None = None

    # pH: float = 7.0
    device: str = 'cpu' # fixed!

    # Internal options
    cutoff_states: int = 4000
    sfreq_cutoff_individual: float = 0.01
    sfreq_cutoff_combined: float = 0.001
    cutoff_export: float = 0.2
    matrix_def: str = 'dG'
    pH_band: float = 10.0

    # Output options
    no_sdf: bool = False
    # write_output: bool = True
    verbose: bool = False

    # pH-scan specific options
    # pH_scan: bool = False
    # pHs: NDArray[np.float64] | None = None

    # def __post_init__(self) -> None:
    #     if self.pHs is None:
    #         self.pHs: NDArray[np.float64] = np.arange(0, 14.1, 0.5)
    #         assert self.pHs is not None

    def run_single(self, pH: float = 7.0) -> None:
        """
        Run the full Autoprot pipeline.

        1. Setup of models, file paths, and preprocessing.
        2. pH-dependent state enumeration and thermodynamic evaluation.
        3. Final analysis, pKa aggregation, and visualization.
        """

        self.pH = pH
        self._setup()
        net_charge, freqs_macro, state_strs_sym, state_freqs_sym, state_qs, indices = self._calc_microstates(self.pH)
        self.prep_single_output(state_strs_sym, state_freqs_sym, state_qs, indices)
        # if save:
        #     mols = [m.mol for m in self.molecule.microstates]
        #     print(mols)
        #     export_sdf(mols, self.name, self.path)#mols: list[Mol], name: str, path_out: str) -> None:

    def run_scan(
            self, 
            pHs: NDArray[np.float64] = np.arange(0, 14.1, 0.5),
            file: str | None = None,
    ) -> None:
        """
        Run pH scan
        """
        self.pHs = pHs
        self._setup()
        self._scan_pH()
        self._finalize_scan(file=file)

    #########################

    # def _single_pH(self) -> None:
        # """ Calculate microstates for single pH value only. """

    def _setup(self) -> None:
        """
        Initialize models, file structure, and neutral-state information.

        This step performs all non-pH-dependent initialization, including:

        - Creating output directories
        - Loading molgpka models
        - Preparing the neutral molecular state
        - Initializing result containers
        - Determining the full set of indices with protonable sites
        """

        if self.verbose:
            print(self.name)
            print(self.smiles, flush=True)

        self.initialize_paths_models_libs()
        self.prepare_neutral_state()

        self.indices0, _ = find_candidate_sites(
                self.base0, self.acid0, self.exclude_base_indices, self.exclude_acid_indices,
                0., pH_band=self.pH_band, verbose=False)
        self.indices0_str = pack_indices(self.indices0)

    def _calc_microstates(self, pH):
        """ Calc microstate frequencies given a pH value """
        indices0_curated, q_options0 = self.calc_curated_indices(pH)

        # Screen coupling between residues
        clusters = self.screen_clusters(indices0_curated, q_options0)
        if self.verbose:
            print(f'Clusters: {clusters}')

        state_freqs_clusters = []
        state_strs_clusters = []
        indices_clusters = []
        
        for c_idx, cluster in enumerate(clusters):

            state_strs_cl, state_freqs_cl, indices_cl = self.process_cluster(cluster, indices0_curated, q_options0, pH)

            state_strs_clusters.append(state_strs_cl)
            state_freqs_clusters.append(state_freqs_cl)
            indices_clusters.append(indices_cl)

        # Inject phosphate clusters:
        if self.phosphate_groups:
            state_strs_poh, state_freqs_poh, oh_ids_poh = calc_phosphate_clusters(self.phosphate_groups,pH,self.matrix_def,
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

        indices_str = pack_indices(indices)
        # Check if indices from clusters have been combined correctly to the full list of indices
        if indices_str != self.indices0_str:
            raise ValueError(f"indices_str {indices_str} not equal indices0_str {self.indices0_str}")

        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        self.construct_mols(state_strs, state_vecs, indices)

        # Symmetry (combine frequencies for chemically identical microstates)
        state_strs_sym, state_freqs_sym = calc_symmetry(
            state_strs, state_freqs_lib, self.mols_libs[indices_str], verbose=self.verbose)

        # Macro-pka properties from combined microstates
        net_charge, state_qs, freqs_macro = calc_macro_props(
            state_strs_sym, state_freqs_sym, self.mols_libs[indices_str])
        
        return net_charge, freqs_macro, state_strs_sym, state_freqs_sym, state_qs, indices

    def _scan_pH(self) -> None:
        """
        Perform the full pH-dependent microstate enumeration and analysis.

        For each pH value in the configured pH grid, this method:

        - Identifies curated candidate titration sites (base on the pH window pH_band)
        - Screens residue coupling and builds clusters of coupled sites
        - Processes each cluster to generate microstates and frequencies
        - Optionally injects phosphate-specific clusters
        - Combines cluster results into global state distributions
        - Applies symmetry reduction
        - Computes macrostate properties (net charge and macro-pKa data)
        - Stores results for later visualization
        - Optionally exports pH-specific outputs
        """

        self.net_charges: list[float] = []
        self.state_freqs_all: dict[str, NDArray[np.float64]] = {}
        self.freqs_macro_all: list[dict[int, float]] = []

        for pH_idx, pH in enumerate(self.pHs.flat):

            net_charge, freqs_macro, state_strs_sym, state_freqs_sym, state_qs, indices = self._calc_microstates(pH)

            self.net_charges.append(net_charge)
            self.freqs_macro_all.append(freqs_macro)

            # Add to results for pH scan
            for state_str, state_freq in zip(state_strs_sym,state_freqs_sym):
                if state_str not in self.state_freqs_all:
                    self.state_freqs_all[state_str] = np.zeros(len(self.pHs))
                self.state_freqs_all[state_str][pH_idx] = state_freq

            # Output pH-specific results for pH
            # if abs(pH - self.pH) < 1e-8:
                # self.prep_single_output(state_strs_sym, state_freqs_sym, state_qs, indices)

    def _finalize_scan(self, file: str | None = None) -> None:
        """
        Post-process results, compute macro-pKa values, and generate outputs.

        This step performs final analysis and visualization, including:

        - Combining and exporting macro-pKa values across pH values
        - Identifying relevant microstates for plotting
        - Generating pH scan plots
        """

        self.net_charges_arr = np.round(np.array(self.net_charges),decimals=4)

        # verbose = True if is_jupyter() else False
        self.pkas_macro = combine_pkas_macro(self.pHs, self.freqs_macro_all)

        # if self.pkas_macro and self.save:
            # export_macro_pkas(self.pkas_macro, self.name, self.outfolder)

        if self.state_freqs_all:
            self.calc_relevant_states()

        # reduce number of microstates for plotting
        # if self.state_freqs_all:
            # N_relevant_states, state_strs_relevant, sfreqs_relevant, mols_relevant, sfreqs_not_relevant = calc_relevant_states(
                    # self.state_freqs_all, self.mols_libs[self.indices0_str], verbose=self.verbose)
            # Plotting of pH scan
            # self.fig_scan = plot_pH_scan(
            #         self.name, self.indices0, self.state_strs_relevant, self.sfreqs_relevant, self.pHs, self.net_charges_arr, 
            #         self.sfreqs_not_relevant, self.pkas_macro, verbose=self.verbose)
            # self.fig_molecules = plot_relevant_states(self.mols_relevant) # self.name, self.outfolder,
            # if self.save:
                # compose_image(self.name, self.N_relevant_states, self.save, self.path)# self.name, self.outfolder)

    def calc_relevant_states(
        self,
        max_states: int = 18,
        ) -> None: 
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

            for state_str, sfreqs in self.state_freqs_all.items():
                if np.max(sfreqs) > cutoff:
                    state_strs_relevant.append(state_str)
                    sfreqs_relevant.append(sfreqs)
                    mols_relevant.append(self.mols_libs[self.indices0_str][state_str])
                    pH_argmaxs.append(np.argmax(sfreqs))
                else:
                    sfreqs_not_relevant.append(sfreqs)
            N_relevant_states = len(state_strs_relevant)
            tries += 1
            cutoff += 0.02

        # Sort by pH value of max freq.
        ps = np.argsort(pH_argmaxs)

        self.N_relevant_states = N_relevant_states
        self.state_strs_relevant = [state_strs_relevant[p] for p in ps]
        self.sfreqs_relevant = [sfreqs_relevant[p] for p in ps]
        self.mols_relevant = [mols_relevant[p] for p in ps]
        self.sfreqs_not_relevant = sfreqs_not_relevant
        if self.verbose:
            print(f'Final N relevant states: {self.N_relevant_states} with cutoff {cutoff}')


    

    #########################

    def initialize_paths_models_libs(self) -> None:
        """
        Initialize output directories, load ML models, and reset internal libraries.
        """

        # if self.write_output:
        # if self.save:
            # os.makedirs(self.path, exist_ok=True)

        # molgpka ML models
        model_file_base: str = f'{ROOT}/weight_base.pth'
        model_file_acid: str = f'{ROOT}/weight_acid.pth'

        self.model_base = load_model(model_file_base,device=self.device)
        self.model_acid = load_model(model_file_acid,device=self.device)

        self.base_libs: dict[str, dict[str, dict[int, float]]] = {} # index_str, state_str, map_idx, value
        self.acid_libs: dict[str, dict[str, dict[int, float]]] = {} # index_str, state_str, map_idx, value
        self.smiles_libs: dict[str, dict[str, str]] = {} # index_str, state_str, smiles_str
        self.mols_libs: dict[str, dict[str, Mol]] = {} # index_str, state_str, rdkit Mol

    def prepare_neutral_state(self) -> None:
        """
        Prepare the neutral reference state of the molecule and compute
        initial pKa predictions.

        This method performs the following steps:

        1. Preprocess the input SMILES into a standardized RDKit molecule.
        2. Identify atom indices to exclude from protonation/deprotonation.
        3. Identify special exception sites (e.g., phosphate groups).
        4. Add explicit hydrogens to the molecule.
        5. Predict base and acid pKa values using the loaded ML models.

        The resulting molecule and prediction dictionaries are stored as
        instance attributes for downstream pipeline steps.
        """

        self.mol0, self.smiles0 = preprocess(self.smiles, verbose=self.verbose)
        self.exclude_base_indices, self.exclude_acid_indices = add_exclusions(self.mol0, verbose=self.verbose)
        self.except_indices, self.phosphate_groups = add_exceptions(self.mol0, verbose=self.verbose)

        if self.verbose:
            print('Processed:')
            print(self.smiles0)
            print(f'Exclude base indices: {self.exclude_base_indices}')
            print(f'Exclude acid indices: {self.exclude_acid_indices}')
            print(f'Except indices: {self.except_indices}')

        if self.phosphate_groups:
            if self.verbose:
                print(self.phosphate_groups)

        mol0_h = Chem.rdmolops.AddHs(self.mol0)

        self.base0, self.acid0 = predict_acid_base(mol0_h,
                                         self.model_base,self.model_acid,
                                         device=self.device,verbose=self.verbose) # returns pkas for map indices

    def calc_curated_indices(self, pH: float) -> tuple[list[int], NDArray[np.int64]]:
        """
        Determine protonation candidate sites and apply exception filtering
        for a given pH value.

        1. Identify possible protonation/deprotonation sites based on
        predicted pKa values and the current pH.
        2. Apply exclusion rules.
        3. Split and remove exception sites (e.g., special functional groups).
        4. Return the curated site indices along with the
        corresponding state options.

        Parameters
        ----------
        pH : float
            Target pH value for state evaluation.

        Returns
        -------
        tuple[list[int], list[int], NDArray[np.int64]]
            - Curated indices after applying exception rules.
            - Corresponding state option matrix (q_options).
        """

        if self.verbose:
            print('='*50)
            print(f'pH: {pH}',flush=True)
        indices0, q_options0 = find_candidate_sites(self.base0, self.acid0, self.exclude_base_indices, self.exclude_acid_indices,
                                                    pH, pH_band=self.pH_band, verbose=False)
        if self.verbose:
            print(f'indices0: {indices0}')
            print(f'q_options0: {q_options0}')
        indices0_curated, q_options0 = split_exceptions(indices0, q_options0, self.except_indices)

        if self.verbose:
            print(f'curated indices0: {indices0_curated}')
            print(f'curated q_options0: {q_options0}')

        return indices0_curated, q_options0

    def process_cluster(
        self,
        cluster: list[int],
        indices0_curated: list[int],
        q_options0: NDArray[np.int64],
        pH: float,
    ) -> tuple[list[str], NDArray[np.float64], list[int]]:
        """ 
        Generate and evaluate microstates for a single protonation cluster at a given pH value.

        Parameters
        ----------
        cluster : list[int]
            Indices (relative to `indices0_curated`) defining the sites
            that belong to the current coupled cluster.
        indices0_curated : list[int]
            Absolute indices of all protonable sites after exclusion
            and exception handling.
        q_options0 : NDArray[np.int64]
            Array of shape (n_sites, 3) encoding protonation options for
            each site:
                0 → deprotonation allowed
                1 → neutral state allowed
                2 → protonation allowed
        pH : float
            Current pH value in the pH scan.

        Returns
        -------
        state_strs : list[str]
            Encoded microstate representations for the cluster.
        state_freqs : NDArray[np.float64]
            Corresponding normalized microstate frequencies,
            ordered identically to `state_strs`.
        indices : list[int]
            Absolute atom map ids (GetAtomMapNum()) corresponding to this cluster.
        """

        indices = [indices0_curated[c] for c in cluster]
        indices_str = pack_indices(indices)

        q_options = q_options0[cluster]
        state_vecs = construct_state_vectors(q_options, self.cutoff_states)

        state_strs = calc_state_strs(state_vecs)

        self.construct_mols(state_strs, state_vecs, indices)
        self.run_acid_base_calcs(state_strs, state_vecs, indices)

        ps_all = calc_state_diffs(
            state_strs, state_vecs, indices,
            self.base_libs[indices_str], self.acid_libs[indices_str], 
            pH=pH, matrix_def=self.matrix_def, verbose=self.verbose)
        
        state_strs, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,self.matrix_def)
        return state_strs, state_freqs, indices

    #########################

    def coupling_assay(self, indices: list[int], q_options: NDArray[np.int64], coupling_cutoff: float) -> list[list[int]]:
        """
        Perform pairwise pKa sensitivity analysis and cluster coupled sites.

        This method evaluates whether protonation of one site affects the
        predicted pKa values of other sites within the provided index set.

        Procedure:
        - Enumerate all single-site protonation states
        - Construct molecular representations for each state
        - Compare pKa values between reference and perturbed states
        - Build a coupling matrix based on pKa differences
        - Cluster sites according to the coupling threshold

        Parameters
        ----------
        indices : list[int]
            Absolute atom map indices of the protonable sites being analyzed.
        q_options : np.ndarray
            Array encoding allowed protonation states for each site.
        coupling_cutoff : float
            Threshold used to determine whether two sites are considered
            coupled based on their pKa differences.

        Returns
        -------
        clusters : list[list[int]]
            List of clusters, where each cluster contains indices of
            mutually coupled sites.
        """

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
                    self.base_libs[indices_str], self.acid_libs[indices_str])
        
        coupling_matrix = construct_coupling_matrix(
                indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, 
                coupling_cutoff)
        clusters = cluster_coupling_matrix(coupling_matrix)
        return clusters

    def screen_clusters(self, indices0: list[int], q_options0: NDArray[np.int64]) -> list[list[int]]:
        """
        Determine stable pKa coupling clusters using adaptive thresholding.

        This method partitions protonable sites into independent clusters
        by repeatedly performing a coupling assay and adjusting the
        coupling threshold until all clusters are computationally stable.

        Stability criterion:
        A cluster is rejected if state enumeration exceeds the allowed
        cutoff, which indicates excessive coupling. In that case,
        the coupling threshold is increased and the clustering is repeated.

        Parameters
        ----------
        indices0 : list[int]
            Absolute atom map indices of candidate protonation sites.
        q_options0 : NDArray[np.int64]
            Array encoding allowed protonation states for each site.

        Returns
        -------
        clusters : list[list[int]]
            Final set of stable coupling clusters.
        """

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

    def construct_mols(self, state_strs: list[str], state_vecs: list[NDArray[np.int64]], indices: list[int]) -> None:
        """
        Construct and cache RDKit molecular objects for protonation states.

        For each unique protonation state defined by `state_strs` and
        `state_vecs`, this method:

        - Builds the corresponding RDKit molecule
        - Assigns the state string as the molecule name
        - Stores the molecule and its SMILES representation

        Molecules are only constructed if they are not already present
        in the cache.

        Parameters
        ----------
        state_strs : list[str]
            Encoded representations of protonation states.
        state_vecs : list[NDArray[np.int64]
            Vector representations corresponding to `state_strs`.
        indices : list[int]
            Absolute atom map indices defining the current cluster.
        """

        indices_str = pack_indices(indices)
        if indices_str not in self.mols_libs:
            self.mols_libs[indices_str] = {}
            self.smiles_libs[indices_str] = {}

        for state_str, state_vec in zip(state_strs, state_vecs):
            if state_str not in self.mols_libs[indices_str]:
                mol_cand, smiles_cand = construct_mol(self.mol0, indices, state_vec)
                mol_cand.SetProp("_Name",state_str)
                self.mols_libs[indices_str][state_str] = mol_cand
                self.smiles_libs[indices_str][state_str] = smiles_cand

    ###################################
    # Acid-base calculation

    def run_acid_base_calcs(
        self,
        state_strs: list[str],
        state_vecs: list[NDArray[np.int64]],
        indices: list[int],
    ) -> None:
        """Compute and cache acid/base pKa predictions for microstates.

        For each protonation state, this method predicts site-specific
        acid and base pKa values using molgpka and stores the
        results in `base_libs` and `acid_libs`.

        Predictions are evaluated from the neutral form of each site:
        - For base predictions, other sites are forced to at least neutral
        (deprotonations ignored).
        - For acid predictions, other sites are forced to at most neutral
        (protonations ignored).

        Parameters
        ----------
        state_strs : list[str]
            Encoded representations of protonation states.
        state_vecs : NDArray[np.int64]
            Protonation state vectors corresponding to `state_strs`.
        indices : list[int]
            Absolute atom map indices defining the current cluster.
        """

        indices_str = pack_indices(indices)
        if indices_str not in self.base_libs:
            self.base_libs[indices_str] = {}
        if indices_str not in self.acid_libs:
            self.acid_libs[indices_str] = {}

        for state_str, state_vec in zip(state_strs,state_vecs):
            if state_str in self.base_libs[indices_str]:
                continue

            if self.verbose:
                print(state_str)

            state_vec_base = np.maximum(state_vec,1) # disregard de-protonations of other sites to assess base probability
            state_str_base = pack_vec(state_vec_base)

            mol_base = self.mols_libs[indices_str][state_str_base]
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

            mol_acid = self.mols_libs[indices_str][state_str_acid]
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

            self.base_libs[indices_str][state_str] = base
            self.acid_libs[indices_str][state_str] = acid

    def prep_single_output(
        self,
        state_strs: list[str],
        state_freqs: list[float],
        state_qs: dict[str, int],
        indices: list[int],
    ) -> None:
        """
        Generate microstate output for the selected pH value.

        This method exports the most relevant protonation states for 
        `pH`. States are selected based on their probability 
        relative to the most populated state.

        The procedure includes:
        - Filtering and sorting microstates by probability
        - Checking stereochemical consistency of generated molecules
        - Exporting results to CSV (and optionally SDF)
        - Producing visualization of the optimal state

        Parameters
        ----------
        state_strs : list[str]
            Encoded microstate representations.
        state_freqs : list[float]
            Corresponding microstate probabilities.
        state_qs : dict[str, int]
            Formal charges associated with each microstate.
        indices : list[int]
            Absolute atom map indices defining the active protonation sites.
        """

        # Max freq
        idx_max = np.argmax(state_freqs)
        state_freq_max = np.max(state_freqs)

        # Select states for pH-specific export
        state_strs_export: list[str] = []
        state_freqs_export_l: list[float] = []

        for state_str, state_freq in zip(state_strs, state_freqs):
            if state_freq > self.cutoff_export * state_freq_max: # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export_l.append(state_freq)

        state_freqs_export: NDArray[np.float64] = np.array(state_freqs_export_l)
        ps = np.argsort(state_freqs_export)[::-1] # Sort by highest probability

        state_freqs_export = state_freqs_export[ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        indices_str = pack_indices(indices)
        self.check_chiral_consistency(state_strs, indices)

        if self.verbose:
            print(f'Export at pH {self.pH}:',flush=True)
            for e_idx, (state_str, sfreq) in enumerate(zip(state_strs_export, state_freqs_export)):
                print(e_idx, state_str, sfreq)

        self.molecule = combine_results(
            self.name, state_strs_export, state_freqs_export, self.mols_libs[indices_str], state_qs)

        # if self.write_output:
        #     export_csv(state_strs_export,self.smiles_out,self.sfreqs_out, state_qs, self.name, self.outfolder, self.fout_csv, self.append_csv)
        #     if not self.no_sdf:
        #         export_sdf(self.mols_out, self.name, self.outfolder)
        #     plot_optimal_state(self.mols_out[0], self.name, self.outfolder)
        # if self.verbose:
        #     print(f'Optimal smiles for pH {self.pH}: {self.smiles_out[0]}')

    def check_chiral_consistency(
        self,
        state_strs: list[str],
        indices: list[int],
    ) -> None:
        """Ensure generated microstate molecules can be embedded consistently.

        This method attempts to generate 3D embeddings for each microstate
        molecule. If embedding fails due to stereochemical constraints,
        chiral tags are removed and embedding is retried.

        Updated molecules and corresponding SMILES strings are written back
        to the internal molecular and SMILES libraries.

        Note that this globally removes chirality when embedding fails!

        Parameters
        ----------
        state_strs : list[str]
            Encoded microstate representations to validate.
        indices : list[int]
            Absolute atom indices defining the current cluster.
        """

        indices_str = pack_indices(indices)
        for state_str in state_strs:
            mol = self.mols_libs[indices_str][state_str]

            mol_h = Chem.AddHs(mol)

            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
            if cid != 0:
                print(f'WARNING: Need to remove chirality for embedding for {state_str}!')
                for atom in mol_h.GetAtoms(): # type: ignore
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED) 
            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore

            if cid != 0:
                raise ValueError(f'{state_str} could not be embedded.')
            for atom in mol.GetAtoms(): # type: ignore
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            
            smiles = Chem.MolToSmiles(mol)
            self.mols_libs[indices_str][state_str] = mol
            self.smiles_libs[indices_str][state_str] = smiles

######################################################################
