# pKasso

pKasso determines protonation states for small molecules based on pKa calculations from a pKa predictor. Currently, MolGpKa (https://github.com/Xundrug/MolGpKa) is used to predict pKa values per sites, with plans to expand to other pKa predictors. 

One protonation microstate describes a unique charge pattern on the protonable sites of molecules. pKasso enumerates protonation microstates, screens pKa values between connected microstates, and predicts pH-dependent microstate frequencies based on the graph of free energy differences between microstates. 

The program runs in different modes: 

1) `single`: Calculate single pH-dependent microstate probabilities given a SMILES string
2) `batch`: Batch process a .smi file to calculate pH-dependent microstates
3) `scan`: Scan a pH range and plot the microstate distributions for all pH values (for a single molecule); calculate macro-pKa values.

# Installation

```
pip install .
```

# Command line interface

Basic example:

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

# Python interface (e.g. in a notebook)

Also see the example jupyter notebook in `example/example.ipynb`

### Single molecule, single pH

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

### Batch molecules (from .smi file)

```
from pkasso import protonate

batch_input = [
    'OC(=O)C(c1ccc(O)cc1)CNCCN',
    'C1CNCCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3',
    'C1=C(NC=N1)CCN',
]

# Use a simple for loop to batch protonate

smiles_batch = []
mols_batch = []

for smiles in batch_input:
    smiles_out, mols_out = protonate(smiles, pH=7., cutoff_export=0.2)
    print(smiles_out)
    smiles_batch.append(smiles_out)
    mols_batch.append(mols_out)

print(smiles_batch)
```

### pH scan (single molecule) in notebook

```
%config InlineBackend.figure_format = 'svg'
from pkasso import scan_pH
from IPython.display import display

smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
name = 'mymolecule'

scan = scan_pH(
    smiles,
    name = name,
)

scan.print_macro_pkas()

display(scan.plot_scan())
display(scan.plot_mols())
```