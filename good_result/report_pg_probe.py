"""Audit saved probe predictions and render categorical classification evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def main():
    raw = (ROOT/'results/pg-probe/534/summary.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '51244f7fb8ed1d7ed2b4a8feb8c11d360bc8fcd0aa562e90c531c246a550a86f'
    data = json.loads(raw)
    y = np.array(data['test_labels'])
    assert len(y) == 677 and len(set(data['test_ids'])) == 677
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
    for ax, key, title in zip(axes,['raw_mean','pooled_norms'],['Raw mean pooling + linear probe','Pooled irrep norms + linear classifier']):
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
    fig.text(.5,.04,'Frozen reduced dielectric split: 677 test structures. Left: strictly linear in pooled features. Right: nonlinear norm preprocessing.',ha='center',fontsize=9)
    for ext in ('png','svg'):
        fig.savefig(OUT/f'pg_probe_confusion.{ext}',dpi=180)
    plt.close(fig)
    print('PASS: saved predictions, confusion matrices, metrics and validation-only alpha selection')


if __name__ == '__main__':
    main()
