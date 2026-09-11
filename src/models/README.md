# Models

`BackboneTensorModel` composes a manifest-pinned checkpoint adapter with the five-branch
downstream dispatcher. The adapter projects into the architecture hidden O(3) layout,
and downstream computation uses the exact edge geometry returned by the backbone.
Tests may inject a contract-compatible extractor only for orchestration tests; final
acceptance always uses all four real adapters through Slurm.
