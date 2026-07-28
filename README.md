# pKasso

Developed by [Bind Research](https://bindresearch.org/)

# Protonation state prediction for small molecules

pKasso determines protonation states for small molecules from SMILES strings or RDKit molecule objects. pKasso is open-source and free to use (MIT Licence).

Protonation microstates describe the unique charge patterns on protonable sites of molecules. The acid/base equilibria (micro-pKa values) of individual protonable sites are coupled, leading to a graph of free energy difference between protonation microstates. pKasso computes this graph based on micro-pKa predictions from [MolGpKa](https://github.com/Xundrug/MolGpKa) (MIT) and optionally standard state free energies from [Uni-pKa](https://github.com/dptech-corp/Uni-pKa) (Apache-2.0). pKasso then converts the results into pH-dependent absolute microstate probabilities and predicts net acid/base equilibria (macro-pKa values) of the molecule.

*pKasso is under active development. Features, prediction models, and results may change in future releases.*

## Local installation

### Basic install (MolGpKa)
```
# Create conda environment
conda create -n pkasso python=3.12
conda activate pkasso

# Install pkasso from PyPI
pip install pkasso
```

### Mixed mode (MolGpKa + Uni-pKa)

pKasso can be run with a mixed model based on MolGpKa and Uni-pKa. In that case, install `pKasso[unipka]`.

```
# Create conda environment
conda create -n pkasso python=3.12
conda activate pkasso

# Install pkasso with Uni-pKa support from PyPI
pip install pkasso[unipka]
```

Mixed mode is more computationally demanding and profits from a GPU. It can run in CPU-only mode as well, albeit more slowly.

### Local webserver

A local webserver can be installed via 
```
pip install pkasso[webserver,unipka]
``` 
followed by calling `pkasso-web` or by downloading and running the [docker image](https://github.com/bindresearch/pkasso/pkgs/container/pkasso) (main). Installing `pkasso[webserver]` without `unipka` does work as well but without 'Precision Mode' (mixed mode) functionality.

## Run pKasso

The easiest way to run pKasso is via the [webserver](https://tools.bindresearch.org/pkasso) hosted by Bind Research.


### Command line interface

The command line interface is called via `pkasso`. 

1) `pkasso single`: Calculate single pH-dependent microstate probabilities given a SMILES string
2) `pkasso batch`: Batch process a .smi file to calculate pH-dependent microstates
3) `pkasso scan`: Scan a pH range and plot the microstate distributions for all pH values (for a single molecule); calculate macro-pKa values.

```
pkasso --smiles 'CCCCNCCCN'
# equivalent to
# pkasso single --smiles 'CCCCNCCCN'
```

Get help for different pKasso options (single prediction, batch prediction, pH scan) with
```
pkasso --help
pkasso single --help
pkasso batch --help
pkasso scan --help
```

All commands accept `--model molgpka` (the default) or `--model mixed`. The mixed model combines MolGpKa with Uni-pKa.

```bash
pkasso single --smiles "CC(=O)O" --model mixed
```

An additional argument `--unipka-model-folder` (defaults to `~/.local/share/unipkainfer/models`) can be provided to define where the Uni-pKa model checkpoint should be stored. A missing checkpoint is downloaded on first use.

### Python interface

```
from pkasso import protonate

name = 'mymolecule'
smiles = r'CCCCNCCCN'
pH = 7.0

# Include microstates with probability of 20% compared to most probable microstate
# Select cutoff_export = 1. to only output the most likely microstate
cutoff_export = 0.2

# protonate accepts a smiles string or an rdkit Mol as input
smiles_out, mols_out = protonate(smiles, name=name, pH=pH, cutoff_export=cutoff_export)
print(smiles_out)
```

Predictors and their options are selected with a dictionary:

```python
model = {
    "molgpka": {},
    "unipka": {
        # "model_dir": "/path/to/unipka/models",
        "folds": (1,),
        "gpu": True,
    },
}
smiles_out, mols_out = protonate(smiles, model=model, pH=pH, nthreads=8)
```

(Ensure that molgpka is the first listed predictor in the dictionary.)

Five variants of Uni-pKa were trained (fold 0 to fold 4). The variants have similar accuracy. By default (including the CLI) fold 1 is used. The selected Uni-pKa model fold is downloaded from Hugging Face on first use and cached for later processes. All model checkpoints are cached in the user data directory. Uncomment the `"model_dir"` keyword to manually define the cache folder.

`nthreads` controls RDKit preprocessing, MolGpKa PyTorch inference, and Uni-pKa preprocessing.

For more examples, see the [jupyter notebook](https://github.com/bindresearch/pkasso/blob/main/example/example.ipynb).
