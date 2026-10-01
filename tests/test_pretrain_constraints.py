"""Compare batched constraints to the pinned source, including derivatives."""
import ast
from pathlib import Path

import pytest
import torch

from src.experiments.pretrain.constraints import equality_adjustment, verify_equivalence


def original_function(task):
    root=Path(__file__).resolve().parents[1]/"data/sources/GMTNet"
    path=root/("GMTNet_elast/gmtnet.py" if task=="elastic" else "gmtnet.py")
    tree=ast.parse(path.read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="equality_adjustment")
    namespace={"torch":torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),namespace)
    return namespace["equality_adjustment"]


@pytest.mark.parametrize("task,width",[("dielectric",3),("elastic",6)])
def test_original_value_and_gradient_equivalence(task,width):
    generator=torch.Generator().manual_seed(118)
    ideal=torch.randint(-3,4,(4,width*width),generator=generator).float()
    equal=(ideal[:,:,None]-ideal[:,None,:]).abs()<1e-5
    opposite=(ideal[:,:,None]+ideal[:,None,:]).abs()<1e-5
    # Zero entries are not marked by the original strict relative-tolerance test.
    nonzero=(ideal[:,:,None].abs()+ideal[:,None,:].abs())>0
    equal &= nonzero
    opposite &= nonzero
    mask=torch.stack((equal,opposite),1) if task=="elastic" else equal
    proof=verify_equivalence(original_function(task),mask,width,device="cpu")
    assert proof["output_max_abs"]<2e-5
    assert proof["gradient_max_abs"]<2e-5


def test_empty_constraint_mask_is_identity_with_identity_gradient():
    value=torch.randn(3,6,6,requires_grad=True)
    result=equality_adjustment(torch.zeros(3,2,36,36,dtype=torch.bool),value)
    torch.testing.assert_close(result,value,atol=0,rtol=0)
    gradient=torch.autograd.grad(result.sum(),value)[0]
    torch.testing.assert_close(gradient,torch.ones_like(value),atol=0,rtol=0)
