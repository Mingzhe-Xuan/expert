"""Converged multinomial logistic probes; train-only scaling and CE + L2."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from .pg_linear import ALPHAS, metrics


def objective(theta, x, y, classes, alpha):
    weight = theta[:x.shape[1]*classes].reshape(x.shape[1],classes)
    bias = theta[x.shape[1]*classes:]
    logits = x @ weight + bias
    logp = logits-logsumexp(logits,axis=1,keepdims=True)
    loss = -logp[np.arange(len(y)),y].mean()+.5*alpha*np.square(weight).sum()
    residual = np.exp(logp)
    residual[np.arange(len(y)),y] -= 1
    residual /= len(y)
    gradient = np.concatenate(((x.T@residual+alpha*weight).ravel(),residual.sum(0)))
    return loss,gradient


def fit_logistic(features, labels, classes, shuffle=False):
    x = {s:np.asarray(v.detach().cpu(),dtype=np.float64) for s,v in features.items()}
    y = {s:np.asarray(v.detach().cpu(),dtype=np.int64) for s,v in labels.items()}
    if any(not np.isfinite(v).all() for v in x.values()):
        raise ValueError('nonfinite input')
    mean,scale = x['train'].mean(0),x['train'].std(0)
    keep = scale > max(scale.max()*1e-8,1e-12)
    if not keep.any():
        raise ValueError('no variable coordinates')
    z = {s:(v[:,keep]-mean[keep])/scale[keep] for s,v in x.items()}
    if shuffle:
        permutation = torch.randperm(len(y['train']),generator=torch.Generator().manual_seed(42)).numpy()
        y['train'] = y['train'][permutation]
    def prediction(theta, split):
        logits = z[split] @ theta[:-classes].reshape(-1,classes)+theta[-classes:]
        return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    def score(theta, split):
        return metrics(torch.from_numpy(y[split]),torch.from_numpy(prediction(theta,split).argmax(1)),classes)
    candidates,solutions = [],{}
    for alpha in ALPHAS:
        result = minimize(objective,np.zeros((int(keep.sum())+1)*classes),
            args=(z['train'],y['train'],classes,alpha),jac=True,method='L-BFGS-B',
            options=dict(maxiter=5000,maxls=50,gtol=1e-8,ftol=1e-14,maxcor=30))
        loss,gradient = objective(result.x,z['train'],y['train'],classes,alpha)
        grad_max = float(np.abs(gradient).max())
        if not result.success or not np.isfinite(loss) or grad_max > 1e-6:
            raise RuntimeError(f'probe did not converge: alpha={alpha}, grad={grad_max}, {result.message}')
        solutions[alpha] = result.x
        candidates.append(dict(alpha=alpha,train=score(result.x,'train'),validation=score(result.x,'validation'),
            convergence=dict(success=True,iterations=int(result.nit),gradient_max=grad_max,objective=float(loss),message=str(result.message))))
    best = max(candidates,key=lambda c:(c['validation']['macro_f1'],c['validation']['accuracy'],c['alpha']))
    theta = solutions[best['alpha']]
    probabilities = prediction(theta,'test')
    report = dict(input_dimension=x['train'].shape[1],retained_dimension=int(keep.sum()),alpha=best['alpha'],
        candidates=candidates,test=score(theta,'test'),test_predictions=probabilities.argmax(1).tolist(),
        test_probabilities=probabilities.tolist())
    state = dict(mean=mean.tolist(),scale=scale.tolist(),keep=keep.tolist(),weight=theta[:-classes].reshape(-1,classes).tolist(),bias=theta[-classes:].tolist())
    return report,state
