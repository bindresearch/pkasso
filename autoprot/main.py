"""Core AutoProt workflow implementation."""

import copy
import itertools
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RegistrationHash
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol

from . import coupling, special_cases, utils
from .external.pka import load_model, predict_acid_base
from .postprocess import combine_results
from .transitions import calc_freqs_from_states, calc_state_diffs
from .utils import pack_indices, pack_vec, unpack_vec
from .tautomers import best_tautomer_smiles

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*") # type: ignore

pkg_base = resources.files('autoprot')

ROOT = f'{pkg_base}/data'

def preprocess(smiles_raw: str, tautomer_search: bool = False) -> tuple[Mol,  str]:
    """ 
    Construct and standardize an RDKit molecule from a SMILES string.
    Charges that cannot be neutralized (e.g., quaternary ammonium) are preserved.
    Atom map numbers are assigned to preserve mapping when the molecule gets changed
    (re-ordered, protonated, de-protonated).

    Parameters
    ----------
    smiles_raw
        Input SMILES string representing the molecule.
    tautomer_search
        Perform rough tautomer search using xtb energies for neutral state

    Returns
    -------
    mol
        The standardized RDKit molecule with atom map numbers set to
        1-based indices.
    smiles
        Canonical SMILES representation of the processed molecule.
    """

    logger.debug('Raw:')
    logger.debug(smiles_raw)
    mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    
    logger.debug('Canonical')
    logger.debug(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    logger.debug('Formal charges before cleanup')
    charges = [at.GetFormalCharge() for at in mol.GetAtoms()] # type: ignore
    logger.debug(charges)

    mol = rdMolStandardize.Cleanup(mol)
    uncharger = rdMolStandardize.Uncharger(force=True)

    # load/save cycles to clean up the mol atom ordering
    mol = uncharger.uncharge(mol)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)

    if tautomer_search:
        smiles = best_tautomer_smiles(smiles)

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)

    for atom in mol.GetAtoms(): # type: ignore
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    logger.debug('Formal charges after cleanup')
    symbols = [at.GetFormalCharge() for at in mol.GetAtoms()] # type: ignore
    logger.debug(symbols)

    return mol, smiles

def find_candidate_sites(
        base: dict[int, float],
        acid: dict[int, float],
        exclude_base_indices: list[int],
        exclude_acid_indices: list[int],
        charged_indices: list[int],
        pH: float,
        pH_band: float = 8.,
) -> tuple[list[int], NDArray[np.int64]]:
    """
    Determine possible protonation and deprotonation sites for a molecule.
    Candidate atom indices are derived from predicted basic and acidic sites.

    Parameters
    ----------
    base
        Mapping of atom map indices to predicted basic pKa values.
    acid
        Mapping of atom map indices to predicted acidic pKa values.
    exclude_base_indices
        Atom map indices that must not be considered for protonation.
    exclude_acid_indices
        Atom map indices that must not be considered for deprotonation.
    charged_indices
        Atom map indices that could not be neutralized
    pH
        Target pH value used to evaluate protonation states.
    pH_band
        Allowed pKa tolerance around the pH when determining candidate
        sites. Default is 8.

    Returns
    -------
    indices
        Sorted atom map indices considered for protonation state changes.
    q_options
        Array of shape (n_sites, 3) indicating allowed states per site:
        [deprotonated, unchanged, protonated].
    """

    prot_candidates = list(base.keys()) # should be map idx
    deprot_candidates = list(acid.keys())

    indices = list(sorted(set(prot_candidates + deprot_candidates)))
    indices_curated: list[int] = []

    logger.debug(f'relevant indices: {indices}')

    for map_idx in indices:
        if map_idx not in charged_indices:
            indices_curated.append(map_idx)

    logger.debug(f'relevant indices (after charged removal): {indices_curated}')

    q_options = np.zeros((len(indices_curated),3),dtype=np.int64) # deprot=0, stay=1, prot=2
    for rel_idx, map_idx in enumerate(indices_curated):
        q_options[rel_idx,1] = 1 # always allow stay
        if map_idx in prot_candidates:
            if map_idx not in exclude_base_indices:
                if base[map_idx] >= pH - pH_band:
                    q_options[rel_idx,2] = 1 # allow protonation
        if map_idx in deprot_candidates:
            if map_idx not in exclude_acid_indices:
                if acid[map_idx] <= pH + pH_band:
                    q_options[rel_idx,0] = 1 # allow deprotonation
    return indices_curated, q_options

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
    q_options
        Array of shape (n_sites, 3) indicating allowed states per site,
        where columns correspond to [deprotonated, unchanged, protonated]
        and entries are 1 (allowed) or 0 (disallowed).
    cutoff_states
        Maximum number of state combinations to enumerate.

    Returns
    -------
    state_vecs
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
        return []
    else:
        state_vecs = [np.array(x) for x in list(itertools.product(*q_options_nonzero))]
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
    mol0
        Reference molecule (neutral standardized structure)
        with atom map numbers assigned.
    indices
        Atom map indices corresponding to the sites whose states are
        defined in ``state_vec``.
    state_vec
        Protonation state vector for the selected sites. Values are encoded
        as [0, 1, 2] corresponding to [deprotonated, unchanged, protonated].

    Returns
    -------
    mol
        RDKit molecule with the specified protonation states applied.
    smiles
        Non-canonical SMILES representation of the constructed molecule.
    """

    mol_cand = copy.deepcopy(mol0)

    smiles_cand = Chem.MolToSmiles(mol_cand, canonical=False)

    qs = state_vec - 1
    
    rw = Chem.RWMol(Chem.AddHs(mol_cand))

    for map_idx, q in zip(indices,qs):
        atom = utils.get_atom_with_map_idx(rw, map_idx)
        if atom is None:
            raise ValueError(f"Could not find atom with map index {map_idx}.")
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
    state_strs_clusters
        Microstate string labels for each cluster.
    state_freqs_clusters
        Corresponding state frequency arrays for each cluster.
    indices_clusters
        Atom indices associated with each cluster (must be non-overlapping).
    sfreq_cutoff_individual
        Minimum frequency required for a state to be considered during
        cluster-wise filtering. Default is 0.01.
    sfreq_cutoff_combined
        Minimum frequency required for a combined microstate to be kept.
        Default is 0.001.

    Returns
    -------
    indices
        Sorted list of all atom indices across clusters.
    state_strs
        Combined microstate string representations passing frequency filters.
    state_freqs_lib
        Dictionary mapping microstate strings to their normalized
        probabilities. Used downstream for calc_symmetry().
    """

    # Cull the state_strs per cluster a bit before combining.
    # This is quite conservative (everything with at least 1% freq in that cluster)

    cluster_state_ids: list[list[int]] = []

    logger.debug(state_strs_clusters)
    logger.debug(state_freqs_clusters)

    for state_freqs in state_freqs_clusters:
        cluster_state_ids.append([])
        for s_idx, s_freq in enumerate(state_freqs):
            if s_freq >= sfreq_cutoff_individual:
                cluster_state_ids[-1].append(s_idx)

    combinations = list(itertools.product(*cluster_state_ids))
    logger.debug(f'N microstate combinations from clusters: {len(combinations)}')

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
        state_str = utils.sort_string(state_str,ps) # match sorted indices                 
        if state_freq >= sfreq_cutoff_combined:
            state_strs.append(state_str)
            state_freqs_list.append(state_freq)
        
    logger.debug(f'N chosen microstate combinations: {len(state_strs)}')
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
    state_strs
        Microstate identifiers used as keys in ``mols_lib``.
    mols_lib
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    hashes
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
) -> tuple[list[str], list[float]]:
    """
    Merge symmetry-equivalent microstates based on molecular hashes.

    Microstates that share the same registration hash are grouped together.
    Their probabilities are summed, and a single representative state
    (alphabetically sorted) is retained per group.

    Parameters
    ----------
    state_strs
        Microstate identifiers.
    state_freqs_lib
        Mapping from microstate strings to their probabilities.
    mols_lib
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    state_strs_symm
        Symmetry-reduced microstate strings.
    state_freqs_symm
        Corresponding combined frequencies.
    """

    state_hashes = calc_hashes(state_strs,mols_lib)
    state_dict: dict[str, list[str]] = {}

    for state_str, state_hash in zip(state_strs,state_hashes):
        if state_hash in state_dict:
            state_dict[state_hash].append(state_str)
        else:
            state_dict[state_hash] = [state_str]

    logger.debug(state_dict)

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
    state_strs
        Microstate identifiers.
    state_freqs
        Corresponding normalized microstate probabilities.
    mols_lib
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    net_charge
        Net charge (frequency-weighted sum)
    state_qs
        Dictionary mapping microstate strings to formal charges
    freqs_macro
        Dictionary mapping formal charges to their total frequency
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
    pHs
        Array of pH values corresponding to the frequency datasets.
    freqs_macro_all
        List of macrostate frequency dictionaries, one per pH value.
        Each dictionary maps formal charge (int) to its frequency.

    Returns
    -------
    pkas_combined
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
    smiles_raw
        Input SMILES string of the molecule to be processed.
    **kwargs
        Optional configuration parameters. Supported keys include:

        Pipeline parameters:
            name, cutoff_states, device, pH_band,
            sfreq_cutoff_individual, sfreq_cutoff_combined,
            matrix_def, cutoff_export

    """

    smiles: str
    name: str = 'molecule'
    
    # Internal options
    cutoff_states: int = 4000
    sfreq_cutoff_individual: float = 0.01
    sfreq_cutoff_combined: float = 0.001
    cutoff_export: float = 0.2
    matrix_def: str = 'dG'
    pH_band: float = 10.0
    device: str = 'cpu' # fixed!
    tautomer_search: bool = False

    def run_single(self, pH: float = 7.0) -> None:
        """
        Run the full Autoprot pipeline.

        1. Setup of models and preprocessing.
        2. pH-dependent state enumeration and thermodynamic evaluation.
        3. Final analysis, pKa aggregation, and visualization.
        """

        self.pH = pH
        self._setup()
        net_charge, freqs_macro, state_strs_sym, state_freqs_sym, state_qs, indices = self._calc_microstates(self.pH)
        self.prep_single_output(state_strs_sym, state_freqs_sym, state_qs, indices)

    def run_scan(
            self, 
            pHs: NDArray[np.float64] = np.arange(0, 14.1, 0.5, dtype=np.float64),
    ) -> None:
        """
        Run pH scan
        """

        self.pHs = pHs
        self._setup()
        self._scan_pH()
        self._finalize_scan()

    #########################

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

        logger.debug(self.name)
        logger.debug(self.smiles, flush=True)

        self.initialize_paths_models_libs()
        self.prepare_neutral_state()

        self.indices0, _ = find_candidate_sites(
                self.base0, self.acid0, self.exclude_base_indices, self.exclude_acid_indices,
                self.charged_indices,
                0., pH_band=100)
        
        # Add indices that should be considered specially but not recognized by molgpka
        for map_idx in self.except_indices:
            if map_idx not in self.indices0:
                self.indices0.append(map_idx)
        
        self.indices0 = list(sorted(self.indices0))
        self.indices0_str = pack_indices(self.indices0)

    def _calc_microstates(self, pH: float
) -> tuple[float, dict[int, float], list[str], list[float], dict[str, int], list[int]]:
        """ Calc microstate frequencies given a pH value """

        indices0_curated, q_options0 = self.calc_curated_indices(pH)

        # Screen coupling between residues
        clusters = self.screen_clusters(indices0_curated, q_options0)
        logger.debug(f'Clusters: {clusters}')

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
            state_strs_poh, state_freqs_poh, oh_ids_poh = special_cases.calc_phosphate_clusters(
                    self.phosphate_groups,pH,self.matrix_def,
            )
            for state_strs, state_freqs, oh_ids in zip(state_strs_poh,state_freqs_poh,oh_ids_poh):
                state_strs_clusters.append(state_strs)
                state_freqs_clusters.append(state_freqs)
                indices_clusters.append(oh_ids)

        if (self.invalid_amine_map_idx > 0) and (self.invalid_amine_map_idx in self.indices0):
            state_strs_invalid_amine, state_freqs_invalid_amine = special_cases.calc_invalid_amine_cluster(pH,self.matrix_def)
            state_strs_clusters.append(state_strs_invalid_amine)
            state_freqs_clusters.append(state_freqs_invalid_amine)
            indices_clusters.append([self.invalid_amine_map_idx])

        # Combine clusters and their frequencies
        indices, state_strs, state_freqs_lib = combine_clusters(
            state_strs_clusters, state_freqs_clusters, indices_clusters, 
            sfreq_cutoff_individual=self.sfreq_cutoff_individual,
            sfreq_cutoff_combined=self.sfreq_cutoff_combined,
        )

        indices_str = pack_indices(indices)
        # Check if indices from clusters have been combined correctly to the full list of indices
        if indices_str != self.indices0_str:
            raise ValueError(f"indices_str {indices_str} not equal indices0_str {self.indices0_str}")

        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        self.construct_mols(state_strs, state_vecs, indices)

        # Symmetry (combine frequencies for chemically identical microstates)
        state_strs_sym, state_freqs_sym = calc_symmetry(
            state_strs, state_freqs_lib, self.mols_libs[indices_str])

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

    def _finalize_scan(self) -> None:
        """
        Post-process results, compute macro-pKa values, and generate outputs.

        This step performs final analysis and visualization, including:

        - Combining and exporting macro-pKa values across pH values
        - Identifying relevant microstates for plotting
        - Generating pH scan plots
        """

        self.net_charges_arr = np.round(np.array(self.net_charges),decimals=4)

        self.pkas_macro = combine_pkas_macro(self.pHs, self.freqs_macro_all)

        if self.state_freqs_all:
            self.calc_relevant_states()

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
        logger.debug(f'Final N relevant states: {self.N_relevant_states} with cutoff {cutoff}')

    #########################

    def initialize_paths_models_libs(self) -> None:
        """
        Initialize output directories, load ML models, and reset internal libraries.
        """

        # molgpka ML models
        model_file_base: Path = f'{ROOT}/weight_base.pth'
        model_file_acid: Path = f'{ROOT}/weight_acid.pth'

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

        self.mol0, self.smiles0 = preprocess(self.smiles, tautomer_search=self.tautomer_search)

        self.charged_indices = special_cases.find_charged(self.mol0)
        self.exclude_base_indices, self.exclude_acid_indices = special_cases.add_exclusions(self.mol0)
        self.except_indices, self.phosphate_groups, self.invalid_amine_map_idx = special_cases.add_exceptions(self.mol0)

        logger.debug('Processed:')
        logger.debug(self.smiles0)
        logger.debug(f'Exclude base indices: {self.exclude_base_indices}')
        logger.debug(f'Exclude acid indices: {self.exclude_acid_indices}')
        logger.debug(f'Except indices: {self.except_indices}')

        if self.phosphate_groups:
            logger.debug(self.phosphate_groups)

        mol0_h = Chem.rdmolops.AddHs(self.mol0)

        self.base0, self.acid0 = predict_acid_base(mol0_h,
                                         self.model_base,self.model_acid,
                                         device=self.device) # returns pkas for map indices

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
        pH
            Target pH value for state evaluation.

        Returns
        -------
        indices0_curated
            Curated indices after applying exception rules.
        q_options0
            Corresponding state option matrix.
        """

        logger.debug('='*50)
        logger.debug(f'pH: {pH}',flush=True)
        indices0, q_options0 = find_candidate_sites(
            self.base0,
            self.acid0,
            self.exclude_base_indices,
            self.exclude_acid_indices,
            self.charged_indices,
            pH,
            pH_band=self.pH_band,
        )
        logger.debug(f'indices0: {indices0}')
        logger.debug(f'q_options0: {q_options0}')
        indices0_curated, q_options0 = special_cases.split_exceptions(indices0, q_options0, self.except_indices)

        logger.debug(f'curated indices0: {indices0_curated}')
        logger.debug(f'curated q_options0: {q_options0}')

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
        cluster
            Indices (relative to `indices0_curated`) defining the sites
            that belong to the current coupled cluster.
        indices0_curated
            Absolute indices of all protonable sites after exclusion
            and exception handling.
        q_options0
            Array of shape (n_sites, 3) encoding protonation options for
            each site:
                0 → deprotonation allowed
                1 → neutral state allowed
                2 → protonation allowed
        pH
            Current pH value in the pH scan.

        Returns
        -------
        state_strs
            Encoded microstate representations for the cluster.
        state_freqs
            Corresponding normalized microstate frequencies,
            ordered identically to `state_strs`.
        indices
            Absolute atom map ids (GetAtomMapNum()) corresponding to this cluster.
        """

        indices = [indices0_curated[c] for c in cluster]
        indices_str = pack_indices(indices)

        q_options = q_options0[cluster]
        state_vecs = construct_state_vectors(q_options, self.cutoff_states)

        state_strs = utils.calc_state_strs(state_vecs)

        self.construct_mols(state_strs, state_vecs, indices)
        self.run_acid_base_calcs(state_strs, state_vecs, indices)

        ps_all = calc_state_diffs(
            state_strs, state_vecs, indices,
            self.base_libs[indices_str], self.acid_libs[indices_str],
            # self.mols_libs[indices_str], 
            pH=pH, matrix_def=self.matrix_def)
        
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
        indices
            Absolute atom map indices of the protonable sites being analyzed.
        q_options
            Array encoding allowed protonation states for each site.
        coupling_cutoff
            Threshold used to determine whether two sites are considered
            coupled based on their pKa differences.

        Returns
        -------
        clusters
            List of clusters, where each cluster contains indices of
            mutually coupled sites.
        """

        indices_str = pack_indices(indices)

        state_vecs = coupling.construct_state_vectors_single(indices, q_options)
        state_strs = utils.calc_state_strs(state_vecs)
        self.construct_mols(state_strs, state_vecs, indices)
        self.run_acid_base_calcs(state_strs, state_vecs, indices)

        state_str0 = state_strs[0] # Neutral state
        base_pka_diffs = {}
        acid_pka_diffs = {}
        for state_str1 in state_strs[1:]:
            base_pka_diffs[state_str1], acid_pka_diffs[state_str1] = coupling.compare_pkas(
                    indices, q_options, state_str0, state_str1, 
                    self.base_libs[indices_str], self.acid_libs[indices_str])
        
        coupling_matrix = coupling.construct_coupling_matrix(
                indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, 
                coupling_cutoff)
        clusters = coupling.cluster_coupling_matrix(coupling_matrix)
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
        indices0
            Absolute atom map indices of candidate protonation sites.
        q_options0
            Array encoding allowed protonation states for each site.

        Returns
        -------
        clusters
            Final set of stable coupling clusters.
        """

        accept_clusters = False
        coupling_cutoff = 0.0
        while not accept_clusters:
            logger.debug(f'coupling cutoff: {coupling_cutoff}')
            accept_clusters = True
            clusters = self.coupling_assay(indices0, q_options0, coupling_cutoff)
            for c_idx, cluster in enumerate(clusters):
                q_options = q_options0[cluster]
                state_vecs = construct_state_vectors(q_options, self.cutoff_states)
                N_states = len(state_vecs)
                if N_states > self.cutoff_states: # Construct_state_vectors should return an empty list
                    raise
                if (N_states == 0):
                    accept_clusters = False
                    coupling_cutoff += 0.2
        if coupling_cutoff > 1.5:
            logger.info(f'Coupling cutoff high: {coupling_cutoff}')
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
        state_strs
            Encoded representations of protonation states.
        state_vecs
            Vector representations corresponding to `state_strs`.
        indices
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
        state_strs
            Encoded representations of protonation states.
        state_vecs
            Protonation state vectors corresponding to `state_strs`.
        indices
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

            logger.debug(state_str)

            state_vec_base = np.maximum(state_vec,1) # disregard de-protonations of other sites to assess base probability
            state_str_base = pack_vec(state_vec_base)

            mol_base = self.mols_libs[indices_str][state_str_base]
            mol_base_h = Chem.rdmolops.AddHs(mol_base)

            base_tmp, _ = predict_acid_base(
                    mol_base_h,self.model_base,self.model_acid,device=self.device,
                    pred_acid=False)
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
                    pred_base=False)
            
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
        state_strs
            Encoded microstate representations.
        state_freqs
            Corresponding microstate probabilities.
        state_qs
            Formal charges associated with each microstate.
        indices
            Absolute atom map indices defining the active protonation sites.
        """

        # Max freq
        state_freq_max = np.max(state_freqs)

        # Select states for pH-specific export
        state_strs_export: list[str] = []
        state_freqs_export: list[float] = []

        for state_str, state_freq in zip(state_strs, state_freqs):
            if state_freq > self.cutoff_export * state_freq_max: # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export.append(state_freq)

        state_freqs_arr: NDArray[np.float64] = np.array(state_freqs_export)
        ps = np.argsort(state_freqs_arr)[::-1] # Sort by highest probability

        state_freqs_export = [state_freqs_export[p] for p in ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        indices_str = pack_indices(indices)
        self.check_chiral_consistency(state_strs, indices)

        logger.debug(f'Export at pH {self.pH}:',flush=True)
        for e_idx, (state_str, sfreq) in enumerate(zip(state_strs_export, state_freqs_export)):
            logger.debug(e_idx, state_str, sfreq)

        self.molecule = combine_results(
            self.name, state_strs_export, state_freqs_export, self.mols_libs[indices_str], state_qs)

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
        state_strs
            Encoded microstate representations to validate.
        indices
            Absolute atom indices defining the current cluster.
        """

        indices_str = pack_indices(indices)
        for state_str in state_strs:
            mol = self.mols_libs[indices_str][state_str]

            mol_h = Chem.AddHs(mol)

            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) # type: ignore
            if cid != 0:
                logger.warning(f'Needed to remove chirality for embedding for {state_str}!')
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
