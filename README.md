# autoprot

Automated protonation state assignment based on pKa calculations from MolGpKa (https://github.com/Xundrug/MolGpKa).

# Installation

```
pip install .
```

# Use

There are several ways to use autoprot.

## Command line interface

```
autoprot --smiles 'OC(=O)C(c1ccc(O)cc1)CNCCN' --name mymolecule
```

Results are output with the most likely microstate first. By default, figures are saved in folder `figures`, output files (including sdf structures and a csv file with a summary) are saved in folder `output`. See `autoprot --help` to change these settings and other behavior.

## Python interface (e.g. in a notebook)

Retrieve rdkit Mol objects, output smiles and frequencies of most likely microstates given a pH value (default: `pH_output = 7.0`).
This does not write any output by default. The behavior can be changed by passing `write_output = True`.

```
from autoprot import protonate

smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'
mols_p, smiles_p, freqs = protonate(smiles, pH=8.5)
```

## Autoprot object

```
from autoprot.main import Autoprot

smiles = r'OC(=O)C(c1ccc(O)cc1)CNCCN'

ap = Autoprot(smiles)
ap.run()

print(ap.smiles_out, ap.sfreqs_out)
# ap.mols_out stores the rdkit molecules
```

This does write output files by default.
