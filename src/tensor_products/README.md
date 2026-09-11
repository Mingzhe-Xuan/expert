# Tensor products

Phase C supplies two interchangeable O(3)-equivariant implementations: `full_o3` and a
complete local-frame `o2_tp`. Placements are adaptation, routed O3E, and final readout.
`o2_tp` defaults to `mmax=2` and retains a full-m control. Both backends share typed
carrier layouts and expose identical input/output shapes.

```python
layer = build_tensor_product("o2_tp", input_layout, edge_layout, output_layout, mmax=2)
```
