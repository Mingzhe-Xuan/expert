import math

import pytest
import torch

from src.training.global_experts.feature_norms import norm_rows, summarize


def test_known_norms_and_mask():
    g = torch.tensor([[3.,4.],[0.,0.]])
    z = torch.tensor([[0.,10.],[1.,2.]])
    rows = norm_rows(g,z,.1)
    assert rows[0]["global_norm"] == 5
    assert rows[0]["scaled_pg_norm"] == 1
    assert rows[0]["scaled_pg_over_global"] == .2
    assert rows[0]["fused_norm"] == math.sqrt(34)
    assert rows[1]["scaled_pg_over_global"] is None
    mask = torch.tensor([[[1.,0.],[0.,0.]]]*2)
    after = norm_rows(g,z,.1,mask)
    assert after[0]["global_norm"] == 3 and after[0]["scaled_pg_norm"] == 0
    stats = summarize(rows)
    assert stats["scaled_pg_over_global"]["undefined"] == 1
    assert stats["scaled_pg_over_global"]["median"] == .2


@pytest.mark.parametrize("failure", ["shape", "nan", "scale", "mask"])
def test_invalid_norm_inputs(failure):
    g = torch.ones(2,3)
    z = g.clone()
    scale, mask = .02, None
    if failure == "shape": z = z[:1]
    if failure == "nan": z[0,0] = float("nan")
    if failure == "scale": scale = float("nan")
    if failure == "mask": mask = torch.ones(1,3,3)
    with pytest.raises(ValueError): norm_rows(g,z,scale,mask)
