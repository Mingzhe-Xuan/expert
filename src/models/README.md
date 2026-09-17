# Models

`BackboneTensorModel` composes a manifest-pinned checkpoint adapter with the five-branch
downstream dispatcher. The adapter projects into the architecture hidden O(3) layout,
and downstream computation uses the exact edge geometry returned by the backbone.
When a native neighbor builder includes an edge exactly on its floating-point cutoff,
the composition layer recomputes lengths from the returned vectors and removes every
edge not satisfying the project's strict `distance < cutoff` convention; it never
widens the physical cutoff.
Tests may inject a contract-compatible extractor only for orchestration tests; final
acceptance always uses all four real adapters through Slurm.

`CGCNNFeatureTensorModel` is an additive non-pretrained branch for controlled comparison with
GMTNet. It consumes the exact fixed 92D JARVIS CGCNN descriptor, applies GMTNet's learned linear
`92 -> 128` even-scalar atom embedding, then projects into the unchanged
`B+A+PGE+R/full_pg` hidden layout. It does not share or replace any DPA4 adapter parameters or
artifacts.
Its forward interface optionally passes a per-sample `ParentDAGSpec` and residual map to the same
downstream dispatcher; absent metadata retains the accepted current-group-only behavior.
