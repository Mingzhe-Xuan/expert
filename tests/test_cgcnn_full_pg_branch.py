from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from src.cli.reduced_cgcnn_full_pg_train import ARCHITECTURE as CGCNN_ARCHITECTURE
from src.cli.reduced_dpa4_train import ARCHITECTURE as DPA4_ARCHITECTURE
from src.cli.reduced_protocol import canonical_expert_point_groups
from src.data import TrainingUnit
from src.features import (
    CGCNN_FEATURE_DIMENSION,
    CGCNN_SOURCE_LAYOUT,
    cgcnn_feature_metadata,
    cgcnn_node_features,
)
from src.models import CGCNNFeatureTensorModel, GMTNET_EMBEDDING_DIMENSION
from src.training.benchmark import _training_loss


def test_cgcnn_features_match_gmtnet_jarvis_provider_and_preserve_order() -> None:
    from jarvis.core.specie import get_node_attributes

    atomic_numbers = torch.tensor([14, 8, 14], dtype=torch.long)
    actual = cgcnn_node_features(atomic_numbers)
    expected = torch.tensor(
        [
            get_node_attributes("Si", atom_features="cgcnn"),
            get_node_attributes("O", atom_features="cgcnn"),
            get_node_attributes("Si", atom_features="cgcnn"),
        ],
        dtype=torch.float32,
    )
    assert actual.shape == (3, CGCNN_FEATURE_DIMENSION)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0], actual[2])
    assert CGCNN_SOURCE_LAYOUT.to_spec() == (
        {
            "multiplicity": 92,
            "degree": 0,
            "parity": "e",
            "copy_label": "gmtnet_cgcnn",
        },
    )
    metadata = cgcnn_feature_metadata()
    assert metadata["dimension"] == 92
    assert len(str(metadata["table_sha256"])) == 64


@pytest.mark.parametrize(
    "atomic_numbers",
    (torch.tensor([], dtype=torch.long), torch.tensor([0]), torch.tensor([104])),
)
def test_cgcnn_features_reject_invalid_atomic_numbers(atomic_numbers) -> None:
    with pytest.raises(ValueError):
        cgcnn_node_features(atomic_numbers)


def test_cgcnn_full_pg_is_additive_and_has_gmtnet_embedding_shape() -> None:
    assert CGCNN_ARCHITECTURE == DPA4_ARCHITECTURE
    model = CGCNNFeatureTensorModel(CGCNN_ARCHITECTURE, "dielectric", ("m-3m",))
    assert model.atom_embedding.in_features == 92
    assert model.atom_embedding.out_features == GMTNET_EMBEDDING_DIMENSION == 128


def test_cgcnn_expert_domain_covers_all_cached_canonical_symmetries() -> None:
    def example(point_group: str) -> SimpleNamespace:
        return SimpleNamespace(
            symmetry=SimpleNamespace(current_point_group=point_group)
        )

    groups = canonical_expert_point_groups(
        (example("m-3m"), example("6/mmm")),
        (example("2/m"),),
        (example("m-3m"),),
    )
    assert groups == ("2/m", "6/mmm", "m-3m")
    model = CGCNNFeatureTensorModel(CGCNN_ARCHITECTURE, "dielectric", groups)
    assert "6/mmm" in model.downstream.expert_point_groups
    with pytest.raises(ValueError, match="every split"):
        canonical_expert_point_groups((example("m-3m"),), ())


def test_gmtnet_protocol_loss_is_raw_cartesian_huber() -> None:
    prediction = SimpleNamespace(
        raw_cartesian=torch.tensor([[[0.0, 2.0], [4.0, 8.0]]]),
        irrep_coefficients=torch.full((1, 2), 999.0),
    )
    batch = SimpleNamespace(
        target_cartesian=torch.tensor([[[0.0, 1.0], [2.0, 4.0]]]),
        target_coefficients=torch.zeros((1, 2)),
    )
    actual = _training_loss(
        prediction,
        batch,
        TrainingUnit("curated_reduced_total", "dielectric"),
        normalizer=None,
        protocol="gmtnet",
    )
    assert torch.equal(actual, F.huber_loss(prediction.raw_cartesian, batch.target_cartesian))
