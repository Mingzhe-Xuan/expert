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
