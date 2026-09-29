from pathlib import Path
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch_geometric.utils import scatter

from src.baselines.gmtnet import GMTNetConfig, build_gmtnet, configure_equivariant_attention
from src.baselines.gmtnet.attention import EquivariantAttentionConv

from e3nn import o3


@pytest.fixture
def official(monkeypatch):
    root = Path(os.environ.get('EXPERT_GMTNET_ROOT',
                Path(__file__).resolve().parents[1] / 'data/sources/GMTNet'))
    if not (root / 'gmtnet.py').is_file():
        if 'EXPERT_GMTNET_ROOT' in os.environ:
            pytest.fail(f'explicit official GMTNet checkout is missing: {root}')
        pytest.skip('optional official GMTNet checkout is unavailable')
    # Exercise the actual official source using the runner's extension-free scatter.
    scatter_module = ModuleType('torch_scatter')
    scatter_module.scatter = scatter
    sparse_module = ModuleType('torch_sparse')
    sparse_module.SparseTensor = object
    monkeypatch.setitem(sys.modules, 'torch_scatter', scatter_module)
    monkeypatch.setitem(sys.modules, 'torch_sparse', sparse_module)
    loaded = {}
    for name in ('utils', 'transformer', 'gmtnet'):
        spec = importlib.util.spec_from_file_location(name, root / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


def test_legacy_and_enabled_full_model(official):
    args = SimpleNamespace(target='dielectric', use_mask=False, reduce_cell=False)
    torch.manual_seed(81)
    original = official.gmtnet.GMTNet(args).eval()
    torch.manual_seed(81)
    default = build_gmtnet(official.gmtnet, args).eval()
    assert GMTNetConfig().use_equiv_attn is False
    assert original.state_dict().keys() == default.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, default.state_dict()[key])
    default.load_state_dict(original.state_dict(), strict=True)
    graph = SimpleNamespace(
        x=torch.randn(4, 92),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 0, 2, 3, 2]]),
        edge_attr=torch.randn(6, 3), batch=torch.zeros(4, dtype=torch.long),
    )
    equality = torch.zeros(1, 9, 9, dtype=torch.bool)
    mask = torch.eye(32).unsqueeze(0)
    expected = original(graph, mask, equality)
    assert torch.equal(default(graph, mask, equality), expected)
    enabled = build_gmtnet(official.gmtnet, args, use_equiv_attn=True).eval()
    missing, unexpected = enabled.load_state_dict(original.state_dict(), strict=False)
    assert missing and all('.attn_mlp.' in key for key in missing)
    assert not unexpected
    assert configure_equivariant_attention(enabled, use_equiv_attn=True) is enabled
    prediction = enabled(graph, mask, equality)
    assert prediction.shape == expected.shape
    args.use_equiv_attn = True
    restored = build_gmtnet(official.gmtnet, args).eval()
    restored.load_state_dict(enabled.state_dict(), strict=True)
    torch.testing.assert_close(restored(graph, mask, equality), prediction)
    prediction.square().sum().backward()
    for name in ('nlayer_1', 'nlayer_2', 'nlayer_3'):
        layer = getattr(enabled.equi_update, name)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.attn_mlp.parameters())
    # Existing DPA4 input replacement remains usable with the option.
    from src.baselines.gmtnet.runner import _replace_atom_embedding
    _replace_atom_embedding(enabled, 6)
    graph.x = torch.randn(4, 6)
    assert enabled(graph, mask, equality).shape == expected.shape


@pytest.mark.parametrize('reflection', [False, True])
def test_attention_o3_weighted_sum_and_empty_edges(official, reflection):
    torch.manual_seed(17)
    layer = EquivariantAttentionConv(official.transformer.TensorProductConvLayer(
        '2x0e + 1x0o + 1x1o', '0e + 1o + 2e', '2x0e + 1x1o', 4,
        residual=False,
    )).double()
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 0, 0, 1, 1], [1, 2, 3, 0, 2]])
    attributes = torch.randn(5, 4, dtype=torch.float64)
    vectors = torch.randn(5, 3, dtype=torch.float64)
    sh = o3.spherical_harmonics(layer.sh_irreps, vectors, True)
    weights = layer.attention_weights(x, edges, attributes)
    assert torch.allclose(scatter(weights, edges[0], dim=0, dim_size=4),
                          torch.tensor([[1.], [1.], [0.], [0.]], dtype=torch.float64))
    messages = layer.tp(x[edges[1]], sh, layer.fc(attributes))
    expected = scatter(weights * messages, edges[0], dim=0, dim_size=4)
    actual = layer(x, edges, attributes, sh)
    torch.testing.assert_close(actual, expected)
    permutation = torch.tensor([3, 0, 4, 2, 1])
    torch.testing.assert_close(
        layer(x, edges[:, permutation], attributes[permutation], sh[permutation]), actual
    )
    q = o3.rand_matrix(dtype=torch.float64) * (-1 if reflection else 1)
    rotated_x = x @ layer.tp.irreps_in1.D_from_matrix(q).T
    rotated_sh = o3.spherical_harmonics(layer.sh_irreps, vectors @ q.T, True)
    torch.testing.assert_close(layer.attention_weights(rotated_x, edges, attributes), weights)
    torch.testing.assert_close(layer(rotated_x, edges, attributes, rotated_sh),
                               actual @ layer.tp.irreps_out.D_from_matrix(q).T,
                               atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.attn_mlp[0].weight.grad.abs().sum() > 0
    empty = layer(x, edges[:, :0], attributes[:0], sh[:0])
    assert empty.shape == (4, 5) and torch.count_nonzero(empty) == 0
    layer.residual = True
    torch.testing.assert_close(layer(x, edges[:, :0], attributes[:0], sh[:0]), x[:, :5])
