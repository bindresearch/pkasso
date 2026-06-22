# pKasso

Developed by [Bind Research](https://bindresearch.org/)

## Protonation state prediction for small molecules

pKasso determines protonation states for small molecules from SMILES strings or RDKit molecule objects. pKasso is open-source and free to use (MIT Licence).

Protonation microstates describe the unique charge patterns on protonable sites of molecules. The acid/base equilibria (micro-pKa values) of individual protonable sites are coupled, leading to a graph of free energy difference between protonation microstates. pKasso computes this graph based on micro-pKa predictions from [MolGpKa](https://github.com/Xundrug/MolGpKa). pKasso then converts the results into pH-dependent absolute microstate probabilities and predicts net acid/base equilibria (macro-pKa values) of the molecule.

*pKasso is under active development. Features, prediction models, and results may change in future releases.*

## Run pKasso

The easiest way to run pKasso is via the [webserver](https://tools.bindresearch.org/pkasso).


### Local installation

```
# Create conda environment
conda create -n pkasso python=3.12
conda activate pkasso

# Install pkasso from PyPI
pip install pkasso
```

### Command line interface

The command line interface is called via `pkasso`. 

1) `pkasso single`: Calculate single pH-dependent microstate probabilities given a SMILES string
2) `pkasso batch`: Batch process a .smi file to calculate pH-dependent microstates
3) `pkasso scan`: Scan a pH range and plot the microstate distributions for all pH values (for a single molecule); calculate macro-pKa values.

```
pkasso --smiles 'OC(=O)C(c1ccc(O)cc1)CNCCN'
# equivalent to
# pkasso single --smiles 'OC(=O)C(c1ccc(O)cc1)CNCCN'
```

Get help for different pKasso options (single prediction, batch prediction, pH scan) with
```
pkasso --help
pkasso single --help
pkasso batch --help
pkasso scan --help
```

### Python interface

```
from pkasso import protonate

name = 'mymolecule'
smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
pH = 7.0

# Include microstates with probability of 20% compared to most probable microstate
# Select cutoff_export = 1. to only output the most likely microstate
cutoff_export = 0.2

# protonate accepts a smiles string or an rdkit Mol as input
smiles_out, mols_out = protonate(smiles, name=name, pH=pH, cutoff_export=cutoff_export)
print(smiles_out)
```

For more examples, see the [jupyter notebook](https://github.com/bindresearch/pkasso/blob/main/example/example.ipynb).

### Local webserver

A local webserver can be hosted via `pip install pkasso[webserver]` followed by calling `pkasso-web` or by downloading and running the [docker image](https://github.com/bindresearch/pkasso/pkgs/container/pkasso) (main).