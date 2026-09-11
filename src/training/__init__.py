"""Loss, optimization, checkpoint, and smoke-training utilities."""

from .normalization import CoefficientNormalizer, NormalizationBlock
from .checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from .losses import CoefficientLoss, coefficient_mse, physical_coefficient_metrics
from .smoke import (
    PreparedTensorBatch,
    build_real_system,
    default_convention_metadata,
    prepare_tensor_batch,
    run_five_structure_smoke,
    write_smoke_report,
)

__all__ = [
    "CoefficientLoss",
    "CoefficientNormalizer",
    "LoadedCheckpoint",
    "NormalizationBlock",
    "PreparedTensorBatch",
    "build_real_system",
    "coefficient_mse",
    "default_convention_metadata",
    "load_checkpoint",
    "physical_coefficient_metrics",
    "prepare_tensor_batch",
    "run_five_structure_smoke",
    "save_checkpoint",
    "write_smoke_report",
]
