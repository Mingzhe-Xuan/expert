"""Independent NumPy audit of four frozen-feature softmax probes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from report_pca_probe import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary',type=Path)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    data = json.loads(raw)
    states = json.loads((args.summary.parent/'classifiers.json').read_text())
    root = Path(__file__).resolve().parents[1]
    assert data['status'] == 'passed' and data['model_optimizer_steps'] == 0
    assert data['split_counts'] == dict(train=5001,validation=637,test=677)
    assert set(data['probes']) == {name+suffix for name in ('dpa_pca32','gmtnet','dpa_gmtnet','global_pg') for suffix in ('','_shuffled')}
    prior = json.loads((Path(__file__).resolve().parents[1]/'results/pca-pg-probe/536/summary.json').read_text())
    assert dict(zip(data['test_ids'],data['test_labels'])) == dict(zip(prior['test_ids'],prior['test_labels']))
    assert data['groups'] == prior['groups'] and data['dataset_sha256'] == prior['dataset_sha256']
    for name,probe in data['probes'].items():
        p = np.array(probe['test_probabilities'])
        assert p.shape == (677,7) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-12)
        assert p.argmax(1).tolist() == probe['test_predictions']
        assert probe['input_dimension'] == 32
        assert len(probe['candidates']) == 7
        for c in probe['candidates']:
            assert c['convergence']['success'] and c['convergence']['gradient_max'] <= 1e-6
        source = data['provenance'][name.removesuffix('_shuffled')]
        path = root/source['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        cache = torch.load(path,map_location='cpu',weights_only=True)
        features = cache['features']['test']
        if isinstance(features,dict):
            features = features[source['tap']]
        index = {key:i for i,key in enumerate(cache['ids']['test'])}
        x = features[[index[key] for key in data['test_ids']]].double().numpy()
        state = states[name]
        keep = np.array(state['keep'],dtype=bool)
        z = ((x-np.array(state['mean']))/np.where(np.array(state['scale'])>0,state['scale'],1))[:,keep]
        logits = z@np.array(state['weight'])+np.array(state['bias'])
        prob = np.exp(logits-logits.max(1,keepdims=True))
        prob /= prob.sum(1,keepdims=True)
        np.testing.assert_allclose(prob,p,rtol=1e-10,atol=1e-12)
    audit(data)
    print('PASS: serialized classifiers reproduce probabilities; convergence, metrics, validation selection and IDs')


if __name__ == '__main__':
    main()
