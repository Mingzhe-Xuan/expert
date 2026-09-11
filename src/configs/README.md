# Configs

`ArchitectureConfig` validates the five frozen branches and requires absent modules to
use `none`. `enumerate_architecture_configs()` returns the exact 26-member matrix mirrored
by `architecture_variant_manifest.json`; invalid combinations fail during construction.

```python
from src.configs import enumerate_architecture_configs

assert len(enumerate_architecture_configs()) == 26
```
