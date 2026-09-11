# Guqq environment locks

These files record independently resolvable acceptance environments. Do not merge backbone locks:
their frozen PyTorch, e3nn, and framework requirements conflict.

`mace-core.txt` is the normalized `pip freeze` from the Guqq MACE/core environment after a clean
`pip check`. The extra index is required for the exact CUDA 12.8 PyTorch build. `python_hostlist`
is the one pure-Python package whose upstream release has no wheel; all native dependencies were
installed from binary wheels.

`grace.txt` records the clean GRACE resolver result. TensorPotential requests TensorFlow's CUDA
extra, whose accepted NVIDIA version ranges are compatible with the exact CUDA 12.8 packages used
by Torch 2.11.0+cu128 in this environment.

`dpa4.txt` records DeepMD-kit 3.2.0 with its Torch extra and the exact CUDA 12.8/e3nn stack used
by the selected DPA4 checkpoint adapter.

`equiformerv2.txt` records fairchem-core 1.10.0 and Torch 2.4.1+cu121. Hydra 1.3.2 and
OmegaConf 2.3.0 are explicit pins: a blanket wheel-only solve otherwise backtracks to obsolete
Hydra 0.11/OmegaConf 1.4. The pure-Python antlr4 runtime is the sole source-build exception.

All four locks come from isolated Guqq environments after clean `pip check` results; do not merge
them or regenerate one from an unexecuted resolver plan.
