# Heads

Task heads map O(3) carriers to dielectric, elastic, or node-wise BEC tensors. The
frozen decompositions are `0e+2e`, `2x0e+2x2e+4e`, and `0e+1e+2e`. BEC retains site
order and may additionally expose ASR and diagnostic symmetry-control tensors without
changing the headline raw output.

```python
prediction = TensorPrediction(raw, coefficients, task="bec", scope="node",
                              node_batch=node_batch)
```

`cartesian_to_irreps` and `irreps_to_cartesian` preserve the frozen repeated-copy
ordering. `apply_bec_asr` is permutation-independent; `project_bec_joint_symmetry` is an
explicit optional diagnostic that requires audited site permutations.
