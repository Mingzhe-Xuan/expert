"""Loss, optimization, checkpoint, and smoke-training utilities."""

from .benchmark import (
    BenchmarkConfig,
    FrozenFeatureBatch,
    FrozenFeatureExample,
    benchmark_target_comparison,
    collate_frozen_examples,
    extract_frozen_examples,
    load_frozen_feature_cache,
    require_published_split_counts,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
)

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
    "BenchmarkConfig",
    "CoefficientLoss",
    "CoefficientNormalizer",
    "FrozenFeatureBatch",
    "FrozenFeatureExample",
    "LoadedCheckpoint",
    "NormalizationBlock",
    "PreparedTensorBatch",
    "build_real_system",
    "benchmark_target_comparison",
    "collate_frozen_examples",
    "coefficient_mse",
    "default_convention_metadata",
    "extract_frozen_examples",
    "load_frozen_feature_cache",
    "require_published_split_counts",
    "load_checkpoint",
    "physical_coefficient_metrics",
    "prepare_tensor_batch",
    "run_five_structure_smoke",
    "save_checkpoint",
    "save_frozen_feature_cache",
    "train_cached_backbone_readout",
    "write_smoke_report",
]
