"""Read-only probe of the trained Global+PG feature fusion, before symmetry mask."""
from dataclasses import fields
import json
from pathlib import Path
import time
import torch

from .pg_linear import DATA_SHA, fit_probe, metrics, pooled_views

SOURCE = Path('results/global-pg-experts/528/summary.json')
SOURCE_SHA = '3b1783acb28f9724c909647692316c7cbe316663f6ae3ea5aaf51888178ade36'
CHECKPOINT_SHA = '61f53fc4cdbf4737b1774160a7f44b6f01a58a81416aaa73e72771d52fd5b57f'
PREDICTIONS_SHA = '93f358cc9ac6ee8e84d811efdd755246a2c5f774a9735673c1750bf8de03a82c'


def feature_taps(diagnostics, mask):
    global_x = diagnostics['global_features'].detach()
    auxiliary = diagnostics['auxiliary'].detach()
    fused = diagnostics['fused_features'].detach()
    torch.testing.assert_close(fused, global_x + diagnostics['branch_scale'].detach()*auxiliary)
    assert fused.ndim == 2 and mask.shape == (len(fused),fused.shape[1],fused.shape[1])
    taps = dict(global_before_mask=global_x, auxiliary_before_mask=auxiliary,
                fused_before_mask=fused, fused_after_mask=torch.bmm(mask,fused.unsqueeze(-1)).squeeze(-1))
    assert all(torch.isfinite(x).all() for x in taps.values())
    return {k:v.cpu() for k,v in taps.items()}


def execute(model, splits, *, output_dir, provenance, config, device):
    from ..training.global_experts.data import file_sha256
    from ..training.global_experts.runner import collate, load_checkpoint, normalized_training_config
    started = time.monotonic()
    assert file_sha256(SOURCE) == SOURCE_SHA
    source = json.loads(SOURCE.read_text())
    checkpoint = SOURCE.parent/'best.pt'
    assert file_sha256(checkpoint) == CHECKPOINT_SHA
    assert file_sha256(SOURCE.parent/'predictions.jsonl') == PREDICTIONS_SHA
    assert source['provenance'] == json.loads(json.dumps(provenance))
    assert source['model_metadata'] == json.loads(json.dumps(model.metadata()))
    model.to(device).eval()
    payload = load_checkpoint(model,checkpoint,provenance=provenance,device=device)
    assert payload['epoch'] == source['best_epoch'] == 196
    assert normalized_training_config(payload['training_config']) == normalized_training_config(source['training_config'])
    assert model.global_model.mask
    # No optimizer or backward step. Official output readout itself needs autograd.
    manifest = json.loads(Path('data/manifests/curated_tensors_reduced_gt_5pct.json').read_text())['artifacts']['dielectric_total']
    assert file_sha256(manifest['path']) == manifest['sha256'] == DATA_SHA
    records = {r['record_id']:r for r in map(json.loads,Path(manifest['path']).read_text().splitlines())}
    groups = manifest['available_point_groups']
    features, targets, ids = {}, {}, {}
    predictions = []
    layout = [dict(multiplicity=mul,degree=ir.l,parity='e' if ir.p==1 else 'o') for mul,ir in model.global_irreps]
    assert model.global_irreps.dim == 32
    for split, rows in splits.items():
        ids[split] = [r['sample_id'] for r in rows]
        assert ids[split] == source['provenance']['split_ids'][split]
        assert set(ids[split]) == {key for key,r in records.items() if r['split']==split}
        targets[split] = torch.tensor([groups.index(records[key]['point_group']) for key in ids[split]])
        collected = {}
        for start in range(0,len(rows),config.batch_size):
            data, mask, equality, routing, _, _ = collate(rows[start:start+config.batch_size],device)
            with torch.enable_grad():
                prediction, diagnostics = model(data,mask,equality,routing,return_diagnostics=True)
            taps = feature_taps(diagnostics,mask)
            taps['fused_pooled_norms'] = torch.stack([pooled_views(x.unsqueeze(0),layout)['pooled_norms'] for x in taps['fused_before_mask']])
            for key, value in taps.items():
                collected.setdefault(key,[]).append(value)
            if split == 'test':
                predictions.append((.5*(prediction+prediction.transpose(-1,-2))).detach().cpu())
            del prediction, diagnostics, data, taps
        features[split] = {k:torch.cat(v) for k,v in collected.items()}
        print(f'extracted {split}: {len(rows)} x32',flush=True)
    assert {s:len(v) for s,v in ids.items()} == dict(train=5001,validation=637,test=677)
    assert len({key for v in ids.values() for key in v}) == 6315
    reference = [json.loads(line) for line in (SOURCE.parent/'predictions.jsonl').read_text().splitlines()]
    assert ids['test'] == [r['sample_id'] for r in reference]
    actual = torch.cat(predictions)
    expected = torch.tensor([r['prediction'] for r in reference],dtype=actual.dtype)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
    reports = {}
    for key in (*features['train'].keys(),'shuffled_fused'):
        name = 'fused_before_mask' if key=='shuffled_fused' else key
        reports[key] = fit_probe({s:f[name] for s,f in features.items()},targets,len(groups),device,key=='shuffled_fused')
        print(key,reports[key]['test'],flush=True)
    majority = int(torch.bincount(targets['train']).argmax())
    assert file_sha256(checkpoint) == CHECKPOINT_SHA
    report = dict(status='passed',source_summary_sha256=SOURCE_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                  checkpoint_epoch=196,source_run=528,dataset_sha256=DATA_SHA,
                  model_optimizer_steps=0,feature_tap='crystal fusion BEFORE explicit symmetry mask',
                  symmetry_conditioned=True,groups=groups,global_irreps=str(model.global_irreps),
                  split_counts={s:len(v) for s,v in ids.items()},test_ids=ids['test'],test_labels=targets['test'].tolist(),
                  majority_baseline=metrics(targets['test'],torch.full_like(targets['test'],majority),len(groups)),
                  branch_scale=float(model.branch_logit.detach().sigmoid()),probes=reports,
                  max_prediction_difference=float((actual-expected).abs().max()),seconds=time.monotonic()-started)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(features=features,ids=ids,labels=targets,layout=layout,checkpoint_sha256=CHECKPOINT_SHA),output_dir/'features.pt')
    return report


def main():
    from ..cli.global_experts_train import parser,run
    from ..models.global_experts import GlobalExpertsConfig
    from ..training.global_experts.runner import normalized_training_config
    from ..training.global_experts.data import file_sha256
    args = parser().parse_args()
    if args.smoke or args.prepare_only or args.global_checkpoint or args.match_pg_active_budget:
        raise ValueError('full frozen checkpoint probe only')
    assert file_sha256(SOURCE) == SOURCE_SHA
    source = json.loads(SOURCE.read_text())
    for key,value in normalized_training_config(source['training_config']).items():
        setattr(args,key,value)
    restored_config = GlobalExpertsConfig(**source['model_metadata']['config'])
    for field in fields(GlobalExpertsConfig):
        setattr(args,field.name,getattr(restored_config,field.name))
    run(args,executor=execute)


if __name__ == '__main__':
    main()
