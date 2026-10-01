import pytest
import torch
from src.probes.pca import project_pca


def test_pca_matches_svd_and_heldout_independence():
    generator = torch.Generator().manual_seed(8)
    splits = {s:torch.randn(n,7,generator=generator,dtype=torch.float64)
              for s,n in [('train',40),('validation',8),('test',9)]}
    projected,state = project_pca(splits,3)
    basis = state['components']
    torch.testing.assert_close(basis.T@basis,torch.eye(3,dtype=torch.float64))
    centered = splits['train']-splits['train'].mean(0)
    _,singular,vh = torch.linalg.svd(centered,full_matrices=False)
    torch.testing.assert_close(basis@basis.T,vh[:3].T@vh[:3])
    torch.testing.assert_close(state['explained_variance_ratio'],singular[:3]**2/(singular**2).sum())
    changed,changed_state = project_pca({**splits,'test':splits['test']*1000,
                                       'validation':splits['validation']+300},3)
    for key in state:
        torch.testing.assert_close(state[key],changed_state[key],rtol=0,atol=0)
    torch.testing.assert_close(projected['train'],changed['train'],rtol=0,atol=0)
    assert {s:tuple(x.shape) for s,x in projected.items()} == {'train':(40,3),'validation':(8,3),'test':(9,3)}
    torch.testing.assert_close(projected['train'].mean(0),torch.zeros(3,dtype=torch.float64),atol=1e-14,rtol=0)


@pytest.mark.parametrize('dimension',[0,8,40])
def test_invalid_dimension(dimension):
    with pytest.raises(ValueError):
        project_pca({'train':torch.ones(40,7)},dimension)


def test_nonfinite_and_rank_rejected():
    with pytest.raises(ValueError,match='nonfinite'):
        project_pca({'train':torch.full((40,7),float('nan'))},3)
    with pytest.raises(ValueError,match='rank'):
        project_pca({'train':torch.ones(40,7)},3)
