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

DPA4 and EquiformerV2 locks must be added only after their isolated Guqq environments pass
`pip check`; do not manufacture them from an unexecuted resolver plan.
