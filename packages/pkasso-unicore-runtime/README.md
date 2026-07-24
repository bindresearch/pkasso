# pkasso-unicore-runtime

`pkasso-unicore-runtime` is an inference-only packaging of the pure-Python
parts of [DP Technology's Uni-Core](https://github.com/dptech-corp/Uni-Core)
needed by pKasso's Uni-pKa model.

It provides an isolated top-level `unicoreinfer` Python package used by
pKasso's vendored Uni-pKa code. Both
CPU and single-GPU inference use ordinary PyTorch operations. The package does
not build or ship Uni-Core's optional fused CUDA extensions, so installing it
does not require a compiler, CUDA toolkit, `nvcc`, or `ninja`.

GPU inference requires a CUDA-enabled PyTorch installation and a compatible
NVIDIA driver. Select the appropriate PyTorch build for the target system
before installing pKasso's `unipka` extra.

## Scope

The runtime retains:

- checkpoint loading;
- argument parsing and registries;
- model, task, and loss base classes;
- dataset and batching utilities;
- transformer and normalization modules;
- logging, metrics, and CUDA tensor movement utilities.

It excludes:

- the training CLI and trainer;
- optimizers and learning-rate schedulers;
- EMA and training diagnostics;
- optional fused C++/CUDA extensions;
- the unused BERT tokenizer adapter.

This package is derived from Uni-Core commit
`ace6fae1c8479a9751f2bb1e1d6e4047427bc134`.
See `unicoreinfer/UPSTREAM.md` for provenance and the included `LICENSE` for
terms.

## Local development

Until the companion distribution is published, install it from its directory
before installing pKasso with the `unipka` extra:

```bash
python -m pip install ./packages/pkasso-unicore-runtime
python -m pip install '.[unipka]'
```

The `unicoreinfer` import namespace is separate from upstream Uni-Core's
`unicore` namespace, so both distributions can coexist in one environment.
