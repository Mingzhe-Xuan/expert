import numpy as np
import torch
from scipy.optimize import check_grad
from src.probes.logistic import objective, fit_logistic


def test_analytic_gradient():
    rng = np.random.default_rng(5)
    x,y,theta = rng.normal(size=(12,4)),np.arange(12)%3,rng.normal(size=15)
    error = check_grad(lambda t:objective(t,x,y,3,.1)[0],lambda t:objective(t,x,y,3,.1)[1],theta)
    assert error < 1e-6


def test_separable_converged_and_heldout_independent():
    labels = {s:torch.arange(n)%3 for s,n in [('train',60),('validation',18),('test',21)]}
    features = {s:torch.cat((torch.nn.functional.one_hot(y,3).double(),torch.ones(len(y),1)),1) for s,y in labels.items()}
    report,state = fit_logistic(features,labels,3)
    assert report['test']['accuracy'] == 1 and report['retained_dimension'] == 3
    np.testing.assert_allclose(np.array(report['test_probabilities']).sum(1),1,atol=1e-12)
    assert all(c['convergence']['gradient_max'] <= 1e-6 for c in report['candidates'])
    changed,_ = fit_logistic({**features,'test':features['test']*10},{**labels,'test':(labels['test']+1)%3},3)
    assert changed['candidates'] == report['candidates'] and changed['alpha'] == report['alpha']
    shuffled,shuffled_state = fit_logistic(features,labels,3,shuffle=True)
    permutation = torch.randperm(60,generator=torch.Generator().manual_seed(42))
    manual,manual_state = fit_logistic(features,{**labels,'train':labels['train'][permutation]},3)
    assert shuffled == manual and shuffled_state == manual_state
    assert shuffled_state != state
    assert state['keep'] == [True,True,True,False]
