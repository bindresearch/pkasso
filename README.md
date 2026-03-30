# Autoprot

Autoprot determines protonation states for small molecules based on pKa calculations from MolGpKa (https://github.com/Xundrug/MolGpKa). 

Autoprot enumerates protonation microstates (one microstate describes a unique charge pattern on the protonable sites of the molecule), screens pKa values between connected microstates and predicts pH-dependent microstate frequencies based on the graph of free energy differences between microstates. 

The program runs in different modes: 

1) Calculate single pH-dependent microstate probabilities given a SMILES string
2) Batch process a .smi file to calculate pH-dependent microstates
3) Screen a pH range and plot the microstate frequencies (for a single molecule); calculate macro-pKa values.

# Installation

```
pip install .
```

# Command line interface

Basic example:

```
autoprot --smiles 'OC(=O)C(c1ccc(O)cc1)CNCCN' --name mymolecule
```
(equivalent to)
```
autoprot single --smiles 'OC(=O)C(c1ccc(O)cc1)CNCCN' --name mymolecule
```

Get help for different autoprot options (single prediction, batch prediction, pH scan) via
```
autoprot --help
autoprot single --help
autoprot batch --help
autoprot scan --help
```

# Python interface (e.g. in a notebook)

Also see the example jupyter notebook in `example/example.ipynb`

### Single molecule, single pH

```
from autoprot import protonate

name = 'mymolecule'
smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
pH = 7.0

# Include microstates with probability of 20% compared to most probable microstate
cutoff_export = 0.2 

molecule = protonate(smiles, name=name, pH=pH, cutoff_export=cutoff_export)
print(molecule.smiles)

molecule.draw()
```

### Batch molecules (from .smi file)

```
from autoprot import batch_protonate

batch_file = 'example_molecules.smi'
batch = batch_protonate(batch_file, pH=7., cutoff_export=0.2)

for name, molecule in batch.molecules.items():
    print(name, molecule.smiles)
```

### pH scan (single molecule) in notebook

```
from autoprot import scan_pH
from IPython.display import display

# smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
name = 'mymolecule'

scan = scan_pH(
    smiles,
    name = name,
)

scan.print_macro_pkas()

display(scan.plot_scan())
display(scan.plot_mols())
```