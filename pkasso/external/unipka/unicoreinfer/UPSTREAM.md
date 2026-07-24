# Upstream provenance

- Project: DP Technology Uni-Core
- Repository: https://github.com/dptech-corp/Uni-Core
- Upstream commit: `ace6fae1c8479a9751f2bb1e1d6e4047427bc134`
- Upstream package version: `0.0.1`
- License: MIT

## Inference-runtime changes

The upstream `unicore` Python package was copied from the pinned commit,
integrated as pKasso's private
`pkasso.external.unipka.unicoreinfer` subpackage, and reduced for Uni-pKa
inference:

- removed `unicore_cli`;
- removed the trainer, optimizers, learning-rate schedulers, EMA, and NaN
  detector;
- omitted all C++ and CUDA extension sources;
- removed eager optimizer imports from `unicoreinfer/__init__.py`;
- rewrote internal imports as package-relative imports and made dynamic module
  loading derive paths from the containing package;
- removed the unused BERT tokenizer adapter and its eager import;
- omitted upstream build metadata and declared the retained runtime's optional
  dependencies in pKasso's `unipka` extra.

Copyright headers in the retained source files are unchanged.
