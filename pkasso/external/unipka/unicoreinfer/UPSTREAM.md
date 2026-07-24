# Upstream provenance

- Project: DP Technology Uni-Core
- Repository: https://github.com/dptech-corp/Uni-Core
- Upstream commit: `ace6fae1c8479a9751f2bb1e1d6e4047427bc134`
- Upstream package version: `0.0.1`
- License: MIT

## Runtime packaging changes

The upstream `unicore` Python package was copied from the pinned commit,
renamed to the isolated `unicoreinfer` import namespace, and reduced for
pKasso Uni-pKa inference:

- removed `unicore_cli`;
- removed the trainer, optimizers, learning-rate schedulers, EMA, and NaN
  detector;
- omitted all C++ and CUDA extension sources;
- removed eager optimizer imports from `unicoreinfer/__init__.py`;
- rewrote internal imports and dynamic module loading from `unicore` to
  `unicoreinfer`;
- removed the unused BERT tokenizer adapter and its eager import;
- replaced upstream build metadata with a PEP 517 `pyproject.toml` for a
  pure-Python wheel;
- reduced dependencies to those imported by the retained runtime.

Copyright headers in the retained source files are unchanged.
