"""Reference forward implementation for the proposal's global tensor model."""

# e3nn 0.4.4 ships trusted tensor constants through ``torch.load``.  PyTorch
# 2.6+ changed that function's default to ``weights_only=True``, which cannot
# deserialize the slice objects in those constants.  MACE also loads its
# official checkpoint as a complete module, so opt into the legacy behavior
# before either dependency is imported.
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from .config import ModelConfig
from .mace_backbone import MACEBackbone, MACEBackboneOutput, MACEPointGroupTensorModel
from .model import CrystalInput, ModelOutput, PointGroupTensorModel

__all__ = [
    "CrystalInput",
    "MACEBackbone",
    "MACEBackboneOutput",
    "MACEPointGroupTensorModel",
    "ModelConfig",
    "ModelOutput",
    "PointGroupTensorModel",
]
