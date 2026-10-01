import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

DATA_SHA = "6cfefc7c04734ceb4c7873e5aaa7d483251ea909a378e55034414d53f9110963"
DPA_SHA = "6820a320dc241a4002e6c257ad21983950df178b1f80271a6e6af199adb18567"
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1.)


def pooled_views(features, layout):
    mean = features.double().mean(0)
    scalars, invariant = [], []
    offset = 0
    for term in layout:
        copies, degree = term['multiplicity'], term['degree']
        width = 2 * degree + 1
        block = mean[offset:offset + copies * width].reshape(copies, width)
        if degree == 0 and term['parity'] == 'e':
            scalars.append(block.flatten())
            invariant.append(block.flatten())
        else:
            invariant.append(block.norm(dim=-1))
        offset += copies * width
    assert offset == mean.numel() and torch.isfinite(mean).all()
    return dict(raw_mean=mean, even_scalars=torch.cat(scalars), pooled_norms=torch.cat(invariant))


def metrics(y, prediction, classes):
    cm = torch.bincount(y.cpu() * classes + prediction.cpu(), minlength=classes**2).reshape(classes, classes).double()
    tp, support = cm.diag(), cm.sum(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + cm.sum(0)).clamp_min(1)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), per_class_recall=recall.tolist(),
                support=support.int().tolist(), confusion_matrix=cm.int().tolist())


def fit_probe(x, y, classes, device='cpu', shuffle=False):
    train = x['train'].to(device=device, dtype=torch.float64)
    mean, scale = train.mean(0), train.std(0, unbiased=False)
    keep = scale > max(float(scale.max()) * 1e-8, 1e-12)
    assert keep.any()
    z = {s: ((v.to(device=device, dtype=torch.float64) - mean)[:, keep] / scale[keep]) for s,v in x.items()}
    labels = y['train'].to(device)
    if shuffle:
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(42)).to(device)
        labels = labels[permutation]
    target = torch.nn.functional.one_hot(labels, classes).double()
    intercept = target.mean(0)
    gram = z['train'].T @ z['train'] / len(train)
    rhs = z['train'].T @ (target - intercept) / len(train)
    values, vectors = torch.linalg.eigh(gram)
    rotated_rhs = vectors.T @ rhs
    candidates = []
    for alpha in ALPHAS:
        weight = vectors @ (rotated_rhs / (values.clamp_min(0)[:, None] + alpha))
        pred = {s: (z[s] @ weight + intercept).argmax(-1).cpu() for s in ('train','validation')}
        candidates.append(dict(alpha=alpha, train=metrics(labels.cpu(), pred['train'], classes),
                               validation=metrics(y['validation'], pred['validation'], classes)))
    best = max(candidates, key=lambda r:(r['validation']['macro_f1'],r['validation']['accuracy'],r['alpha']))
    weight = vectors @ (rotated_rhs / (values.clamp_min(0)[:, None] + best['alpha']))
    prediction = (z['test'] @ weight + intercept).argmax(-1).cpu()
    return dict(input_dimension=train.shape[1], retained_dimension=int(keep.sum()),
                alpha=best['alpha'], candidates=candidates, test=metrics(y['test'],prediction,classes),
                test_predictions=prediction.tolist())


def run(output, device):
    started = time.monotonic()
    root = Path.cwd()
    manifest = json.loads((root/'data/manifests/curated_tensors_reduced_gt_5pct.json').read_text())['artifacts']['dielectric_total']
    data_path = root / manifest['path']
    raw = data_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256'] == DATA_SHA
    records = [json.loads(line) for line in raw.splitlines()]
    rows = {r['record_id']:r for r in records}
    assert len(rows) == len(records) == 6315
    groups = manifest['available_point_groups']
    assert dict(Counter(r['point_group'] for r in records)) == manifest['point_group_counts']
    split_ids, matrices, targets, cached_agreement = {}, {}, {}, {}
    seen = set()
    layouts = []
    for split, count in [('train',5001),('validation',637),('test',677)]:
        ids, labels, views, agreement = [], [], {}, 0
        for shard in range(64):
            path = root/f'results/reduced-benchmark/cache/dpa4/curated_reduced_total__dielectric/full/shards-64/shard-{shard}/{split}.pt'
            cache = torch.load(path, map_location='cpu', weights_only=True)
            assert cache['schema_version'] == 2 and cache['backbone'] == 'dpa4'
            assert cache['dataset_sha256'] == DATA_SHA and cache['checkpoint_sha256'] == DPA_SHA
            assert cache['split'] == f'{split}:shard:{shard}/64'
            assert cache['sample_ids'] == [r['sample_id'] for r in cache['examples']]
            if layouts:
                assert layouts[0] == cache['layout']
            else:
                layouts.append(cache['layout'])
            for entry in cache['examples']:
                key = entry['sample_id']
                assert key not in seen and rows[key]['split'] == split
                seen.add(key)
                ids.append(key)
                labels.append(groups.index(rows[key]['point_group']))
                agreement += entry['symmetry']['current_point_group'] == rows[key]['point_group']
                features = pooled_views(entry['features'], cache['layout'])
                composition = torch.bincount(entry['graph']['atomic_numbers'], minlength=119)[1:].double()
                features['composition'] = composition / composition.sum()
                for name, value in features.items():
                    views.setdefault(name, []).append(value)
            del cache
        assert len(ids) == count and set(ids) == {r['record_id'] for r in records if r['split']==split}
        split_ids[split], targets[split] = ids, torch.tensor(labels)
        matrices[split] = {k:torch.stack(v) for k,v in views.items()}
        cached_agreement[split] = dict(agree=agreement,total=count)
        print(f'pooled {split}: {count}', flush=True)
    majority = int(torch.bincount(targets['train'], minlength=len(groups)).argmax())
    results = {}
    for name in ('raw_mean','even_scalars','pooled_norms','composition','shuffled_raw_mean'):
        feature = 'raw_mean' if name == 'shuffled_raw_mean' else name
        results[name] = fit_probe({s:matrices[s][feature] for s in matrices}, targets, len(groups), device, name=='shuffled_raw_mean')
        print(name, results[name]['alpha'], results[name]['test'], flush=True)
    summary = dict(status='passed', classifier='affine_multiclass_ridge_one_hot',
                   label_definition='source_point_group', groups=groups, dataset_sha256=DATA_SHA,
                   dpa_checkpoint_sha256=DPA_SHA, layout=layouts[0], canonical_frame_caveat=True,
                   split_counts={s:len(v) for s,v in split_ids.items()},
                   class_counts={s:torch.bincount(v,minlength=len(groups)).tolist() for s,v in targets.items()},
                   cached_current_pg_agreement=cached_agreement, selection='validation_macro_f1_then_accuracy_then_larger_alpha',
                   majority_baseline=metrics(targets['test'], torch.full_like(targets['test'], majority), len(groups)),
                   probes=results, test_ids=split_ids['test'], test_labels=targets['test'].tolist(),
                   seconds=time.monotonic()-started, torch_version=str(torch.__version__))
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output,args.device)
