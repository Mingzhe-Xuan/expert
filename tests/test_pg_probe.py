import torch
from src.probes.pg_linear import pooled_views, fit_probe, metrics


def test_pooling():
    layout = [dict(multiplicity=2,degree=0,parity='e'),dict(multiplicity=1,degree=1,parity='o')]
    x = torch.arange(15.).reshape(3,5)
    a, b = pooled_views(x,layout), pooled_views(x.flip(0),layout)
    for key in a:
        torch.testing.assert_close(a[key],b[key])
    torch.testing.assert_close(a['raw_mean'],x.double().mean(0))
    assert a['even_scalars'].shape == (2,) and a['pooled_norms'].shape == (3,)


def test_linear_separability_and_test_independence():
    torch.manual_seed(7)
    y = {s:torch.arange(n)%3 for s,n in [('train',60),('validation',18),('test',21)]}
    x = {s:torch.cat((torch.nn.functional.one_hot(v,3).double(),torch.ones(len(v),1)),1) for s,v in y.items()}
    result = fit_probe(x,y,3)
    assert result['test']['accuracy'] == 1 and result['retained_dimension'] == 3
    changed = fit_probe({**x,'test':x['test']*100}, {**y,'test':(y['test']+1)%3},3)
    assert result['alpha'] == changed['alpha'] and result['candidates'] == changed['candidates']


def test_metrics():
    result = metrics(torch.tensor([0,0,1]),torch.tensor([0,1,1]),2)
    assert result['confusion_matrix'] == [[1,1],[0,1]]
    assert abs(result['accuracy']-2/3)<1e-10
