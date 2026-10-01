"""Audit saved probe predictions and render categorical classification evidence."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fused-summary', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    fused = args.fused_summary is not None
    if fused and not args.sha256:
        parser.error('--fused-summary requires independently verified --sha256')
    raw = (args.fused_summary if fused else ROOT/'results/pg-probe/534/summary.json').read_bytes()
    expected_sha = args.sha256 if fused else '51244f7fb8ed1d7ed2b4a8feb8c11d360bc8fcd0aa562e90c531c246a550a86f'
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    data = json.loads(raw)
    y = np.array(data['test_labels'])
    assert len(y) == 677 and len(set(data['test_ids'])) == 677
    if fused:
        previous = json.loads((ROOT/'results/pg-probe/534/summary.json').read_text())
        assert dict(zip(data['test_ids'],data['test_labels'])) == dict(zip(previous['test_ids'],previous['test_labels']))
        assert data['status'] == 'passed' and data['checkpoint_epoch'] == 196
        assert data['model_optimizer_steps'] == 0 and data['symmetry_conditioned']
    for name, probe in data['probes'].items():
        pred = np.array(probe['test_predictions'])
        cm = np.bincount(y*7+pred,minlength=49).reshape(7,7)
        assert cm.tolist() == probe['test']['confusion_matrix']
        assert abs(np.trace(cm)/677-probe['test']['accuracy']) < 1e-12
        f1 = (2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)).mean()
        assert abs(f1-probe['test']['macro_f1']) < 1e-12
        best = max(probe['candidates'],key=lambda r:(r['validation']['macro_f1'],r['validation']['accuracy'],r['alpha']))
        assert best['alpha'] == probe['alpha']
        print(name, f"accuracy={100*probe['test']['accuracy']:.2f}% macroF1={100*f1:.2f}%")
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    keys = ['global_before_mask','fused_before_mask'] if fused else ['raw_mean','pooled_norms']
    titles = ['Same-checkpoint global features (32D)','Global + PG fused features (32D)'] if fused else ['Raw mean pooling + linear probe','Pooled irrep norms + linear classifier']
    for ax, key, title in zip(axes,keys,titles):
        cm = np.array(data['probes'][key]['test']['confusion_matrix'])
        fraction = cm / cm.sum(1,keepdims=True)
        im = ax.imshow(fraction, vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=range(7),xticklabels=data['groups'],yticks=range(7),yticklabels=data['groups'],
               xlabel='Predicted point group',ylabel='Source point group',title=title)
        for i in range(7):
            for j in range(7):
                ax.text(j,i,f'{cm[i,j]}\n{fraction[i,j]:.0%}',ha='center',va='center',fontsize=8,
                        color='white' if fraction[i,j]>.5 else 'black')
    fig.subplots_adjust(top=.85,bottom=.17,left=.08,right=.91,wspace=.32)
    cax=fig.add_axes([.93,.21,.015,.59])
    fig.colorbar(im,cax=cax,label='Fraction within true class')
    caption = ('Frozen reduced dielectric split: 677 test structures. Both before explicit mask. PG routing already uses point-group information.'
               if fused else 'Frozen reduced dielectric split: 677 test structures. Left: strictly linear in pooled features. Right: nonlinear norm preprocessing.')
    fig.text(.5,.04,caption,ha='center',fontsize=9)
    for ext in ('png','svg'):
        prefix = 'fused_pg_probe_confusion' if fused else 'pg_probe_confusion'
        fig.savefig(OUT/f'{prefix}.{ext}',dpi=180)
    plt.close(fig)
    print('PASS: saved predictions, confusion matrices, metrics and validation-only alpha selection')


if __name__ == '__main__':
    main()
