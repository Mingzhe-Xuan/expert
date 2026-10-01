"""Independently audit a PCA probe and its fixed Job535 branch comparators."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(data):
    y = np.array(data['test_labels'])
    assert len(y) == len(set(data['test_ids'])) == 677
    for name, probe in data['probes'].items():
        pred = np.array(probe['test_predictions'])
        cm = np.bincount(7*y+pred,minlength=49).reshape(7,7)
        assert cm.tolist() == probe['test']['confusion_matrix']
        assert abs(cm.trace()/len(y)-probe['test']['accuracy']) < 1e-12
        f1 = np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1))
        assert abs(f1-probe['test']['macro_f1']) < 1e-12
        best = max(probe['candidates'],key=lambda c:(c['validation']['macro_f1'],c['validation']['accuracy'],c['alpha']))
        assert best['alpha'] == probe['alpha']
        print(name, 'dims', probe['input_dimension'],probe['retained_dimension'],
              'alpha',probe['alpha'],'train',best['train']['accuracy'],
              'val',best['validation']['accuracy'],'test',probe['test']['accuracy'],'f1',f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary',type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--gmtnet-summary',type=Path)
    parser.add_argument('--gmtnet-sha256')
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    pca = json.loads(raw)
    root = Path(__file__).resolve().parents[1]
    raw = (root/'results/fused-pg-probe/535/summary.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '64b110cd63c84e7c0f86808cde3df72b0f0ae9b9c030db00fe77b74b1231cdae'
    fused = json.loads(raw)
    assert pca['groups'] == fused['groups'] and pca['dataset_sha256'] == fused['dataset_sha256']
    assert dict(zip(pca['test_ids'],pca['test_labels'])) == dict(zip(fused['test_ids'],fused['test_labels']))
    assert pca['split_counts'] == fused['split_counts'] == dict(train=5001,validation=637,test=677)
    assert pca['pca']['fit_split'] == 'train' and pca['pca']['output_dimension'] == 32
    ratio = np.array(pca['pca']['explained_variance_ratio'])
    assert ratio.shape == (32,) and np.all(ratio >= 0) and np.all(np.diff(ratio) <= 0)
    assert abs(ratio.sum()-pca['pca']['cumulative_explained_variance']) < 1e-12
    assert 0 < ratio.sum() <= 1
    audit(pca)
    audit(fused)
    if args.gmtnet_summary:
        if not args.gmtnet_sha256:
            parser.error('--gmtnet-summary requires --gmtnet-sha256')
        raw = args.gmtnet_summary.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == args.gmtnet_sha256
        gmtnet = json.loads(raw)
        assert gmtnet['source_run'] == 498 and gmtnet['checkpoint_epoch'] == 196
        assert gmtnet['status'] == 'passed' and gmtnet['model_optimizer_steps'] == 0
        assert gmtnet['groups'] == pca['groups'] and gmtnet['dataset_sha256'] == pca['dataset_sha256']
        assert dict(zip(gmtnet['test_ids'],gmtnet['test_labels'])) == dict(zip(pca['test_ids'],pca['test_labels']))
        audit(gmtnet)
    print('PCA cumulative explained variance',ratio.sum())
    print('PASS: all metrics, validation choices and test-ID labels agree')


if __name__ == '__main__':
    main()
