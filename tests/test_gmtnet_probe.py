import pytest
import torch
from src.probes.gmtnet_pg import pool_nodes


def test_pool_detached_and_mask_distinct():
    nodes = torch.arange(12.).reshape(3,4).requires_grad_()
    pooled = pool_nodes(nodes,torch.tensor([0,0,1]))
    torch.testing.assert_close(pooled,torch.stack((nodes[:2].mean(0),nodes[2])))
    assert not pooled.requires_grad and nodes.grad is None
    mask = torch.eye(4).repeat(2,1,1)
    mask[:,0,0] = 0
    masked = torch.bmm(mask,pooled.unsqueeze(-1)).squeeze(-1)
    assert (masked[:,0]==0).all() and (pooled[:,0]!=0).all()
    with pytest.raises(ValueError,match='noncontiguous'):
        pool_nodes(nodes,torch.tensor([0,0,2]))
