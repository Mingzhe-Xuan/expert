"""Batch the original ordered equality adjustments without changing their sequence."""
from __future__ import annotations

import torch


def equality_adjustment(equality, values):
    shape=values.shape
    flat=values.reshape(shape[0],-1)
    # One mask transfer replaces thousands of Python predicates on CUDA scalars.
    mask_cpu=equality.detach().cpu()
    if equality.ndim==3:
        active_pairs=torch.nonzero(torch.triu(mask_cpu.any(dim=0),diagonal=1),as_tuple=False).tolist()
        for j,k in active_pairs:
            active=equality[:,j,k]
            mean=(flat[:,j]+flat[:,k])*.5
            updated=flat.clone()
            updated[:,j]=torch.where(active,mean,flat[:,j])
            updated[:,k]=torch.where(active,mean,flat[:,k])
            flat=updated
    elif equality.ndim==4 and equality.shape[1]==2:
        positive=equality[:,0]
        for j in torch.nonzero(mask_cpu[:,0].any(dim=(0,2)),as_tuple=False).flatten().tolist():
            active=positive[:,j]
            count=active.sum(-1).clamp_min(1)
            mean=(flat*active).sum(-1)/count
            flat=torch.where(active,mean[:,None],flat)
        negative=equality[:,1]
        active_pairs=torch.nonzero(torch.triu(mask_cpu[:,1].any(dim=0),diagonal=1),as_tuple=False).tolist()
        for j,k in active_pairs:
            active=negative[:,j,k]
            magnitude=(flat[:,j]-flat[:,k]).abs()*.5
            value=torch.where(flat[:,j]<0,-magnitude,magnitude)
            updated=flat.clone()
            updated[:,j]=torch.where(active,value,flat[:,j])
            updated[:,k]=torch.where(active,-value,flat[:,k])
            flat=updated
    else:
        raise ValueError("unexpected equality mask shape")
    return flat.reshape(shape)


def verify_equivalence(original, masks, width, *, device):
    """Compare values and derivatives on actual structure masks before opt-in."""
    generator=torch.Generator(device=device).manual_seed(901)
    first=torch.randn((len(masks),width,width),generator=generator,device=device,requires_grad=True)
    second=first.detach().clone().requires_grad_(True)
    reference=original(masks,first.clone())
    actual=equality_adjustment(masks,second.clone())
    weights=torch.randn(reference.shape,generator=generator,device=device)
    ref_grad=torch.autograd.grad((reference*weights).sum(),first)[0]
    actual_grad=torch.autograd.grad((actual*weights).sum(),second)[0]
    torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(actual_grad,ref_grad,atol=2e-5,rtol=2e-5)
    return {"output_max_abs":float((actual-reference).abs().max()),
            "gradient_max_abs":float((actual_grad-ref_grad).abs().max()),"sample_count":len(masks)}
