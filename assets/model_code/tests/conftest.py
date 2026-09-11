"""Test-process compatibility for MACE's e3nn 0.4.4 dependency."""

import os


os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
