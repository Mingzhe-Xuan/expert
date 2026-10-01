"""Unsupervised centered PCA fitted exclusively on training structures."""
import torch


def project_pca(splits, dimension=32, device='cpu'):
    train = splits['train'].to(device=device, dtype=torch.float64)
    if train.ndim != 2 or not 1 <= dimension <= min(train.shape[0]-1, train.shape[1]):
        raise ValueError('invalid PCA dimension or training matrix')
    if any(x.ndim != 2 or x.shape[1] != train.shape[1] or not torch.isfinite(x).all()
           for x in splits.values()):
        raise ValueError('inconsistent or nonfinite feature matrix')
    mean = train.mean(0)
    centered = train - mean
    covariance = centered.T @ centered / (len(train)-1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    variance = eigenvalues.flip(0).clamp_min(0)
    if variance.sum() <= 0 or variance[dimension-1] <= variance[0]*1e-12:
        raise ValueError('insufficient training rank for requested PCA dimension')
    components = eigenvectors[:, -dimension:].flip(1)
    # Fix sign ambiguity for reproducible serialized coordinates.
    signs = components.gather(0, components.abs().argmax(0, keepdim=True)).sign()
    components = components * signs
    projected = {s:((x.to(device=device,dtype=torch.float64)-mean) @ components).cpu()
                 for s,x in splits.items()}
    state = dict(mean=mean.cpu(), components=components.cpu(),
                 explained_variance=variance[:dimension].cpu(),
                 explained_variance_ratio=(variance[:dimension]/variance.sum()).cpu())
    return projected, state
