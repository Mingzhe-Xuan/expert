# Configs

`ArchitectureConfig` validates the five frozen branches and requires absent modules to
use `none`. `enumerate_architecture_configs()` returns the exact 26-member matrix mirrored
by `architecture_variant_manifest.json`; invalid combinations fail during construction.

```python
from src.configs import enumerate_architecture_configs

assert len(enumerate_architecture_configs()) == 26
```

`real_smoke_schedule.json` freezes 20 independent 3/1/1 runs: every training unit uses
all five branches; every target sees all four backbones; and both PG modes plus both
backends at every actual O(3) TP placement are covered across the schedule.
