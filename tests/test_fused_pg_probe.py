import pytest
import torch
from src.probes.fused_pg import feature_taps


def test_pre_mask_fusion_and_detach():
    g = torch.tensor([[1.,2.,3.]],requires_grad=True)
    z = torch.tensor([[2.,4.,6.]],requires_grad=True)
    scale = torch.tensor(.25)
    diag = dict(global_features=g,auxiliary=z,branch_scale=scale,fused_features=g+scale*z)
    mask = torch.diag(torch.tensor([1.,0.,0.])).unsqueeze(0)
    taps = feature_taps(diag,mask)
    torch.testing.assert_close(taps['fused_before_mask'],torch.tensor([[1.5,3.,4.5]]))
    torch.testing.assert_close(taps['fused_after_mask'],torch.tensor([[1.5,0.,0.]]))
    assert all(not x.requires_grad for x in taps.values())
    assert g.grad is None and z.grad is None
    with pytest.raises(AssertionError):
        feature_taps({**diag,'fused_features':g},mask)
