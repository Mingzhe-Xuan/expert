"""Frozen standalone DPA-GMTNet498 final-carrier PG probe."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch

from .pg_linear import DATA_SHA, DPA_SHA, fit_probe

SOURCE = Path('results/reduced-benchmark/dpa4-gmtnet/summary-498.json')
SOURCE_SHA = '4e554775dacd000297f8b6ca0fcaf1b2fe3c522785302c33751dc0201c176b31'
CHECKPOINT_SHA = 'db0af116def62c239e79b829686e4592eaf0f4d905871bc80687230ffc51a457'
PREDICTIONS_SHA = 'de9c44069b4e595b0a8c24d9ce7cc13b1ac4ed2b3382136fdf05fe0ac69e0f42'


def pool_nodes(nodes, batch):
    nodes = nodes.detach()
    count = int(batch.max())+1
    if nodes.ndim != 2 or batch.shape != (len(nodes),) or not torch.isfinite(nodes).all():
        raise ValueError('invalid nodes or batch')
    counts = torch.bincount(batch,minlength=count)
    if not (counts > 0).all():
        raise ValueError('noncontiguous batch')
    return nodes.new_zeros(count,nodes.shape[1]).index_add_(0,batch,nodes)/counts[:,None]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    from ..data import TrainingUnit, load_training_dataset
    from ..cli.reduced_dpa4_gmtnet_train import _load_dpa4_feature_splits
    from ..baselines.gmtnet import GMTNET_OFFICIAL_COMMIT
    from ..baselines.gmtnet.runner import (_prepare_cache,_attach_dpa4_node_embeddings,
        _load_official_modules,_replace_atom_embedding,_collate)
    from torch_geometric.data import Batch
    started = time.monotonic()
    assert sha(SOURCE) == SOURCE_SHA
    source = json.loads(SOURCE.read_text())
    checkpoint = Path(source['checkpoint'])
    assert sha(checkpoint) == CHECKPOINT_SHA and sha(source['predictions']) == PREDICTIONS_SHA
    assert source['dataset_sha256'] == DATA_SHA and source['dpa4_checkpoint_sha256'] == DPA_SHA
    unit = TrainingUnit('curated_reduced_total','dielectric')
    manifest = Path('data/manifests/curated_tensors_reduced_gt_5pct.json')
    dataset = load_training_dataset(unit,manifest_path=manifest)
    assert dataset[0].source['manifest_sha256'] == DATA_SHA
    ids = {s:tuple(getattr(dataset.split_manifest,s)) for s in ('train','validation','test')}
    assert {s:len(v) for s,v in ids.items()} == dict(train=5001,validation=637,test=677)
    layout,features = _load_dpa4_feature_splits(cache_root=Path('results/reduced-benchmark/cache'),
        unit=unit,full_ids=ids,selected_ids=ids,shard_count=64,checkpoint_sha256=DPA_SHA,dataset_sha256=DATA_SHA)
    splits = _prepare_cache(dataset,args.official_root,Path('results/reduced-benchmark/cache/gmtnet-graphs.pt'),DATA_SHA,ids)
    splits = _attach_dpa4_node_embeddings(splits,features,layout)
    del features
    official,graphs,_ = _load_official_modules(args.official_root)
    torch.manual_seed(source['config']['seed'])
    model = official.GMTNet(SimpleNamespace(target='dielectric',use_mask=True,reduce_cell=False))
    _replace_atom_embedding(model,640)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['epoch'] == source['best_epoch'] == 196
    assert saved['official_commit'] == source['official_commit'] == GMTNET_OFFICIAL_COMMIT
    assert saved['dataset_sha256'] == DATA_SHA
    assert saved['input_embedding'] == source['input_embedding']
    assert saved['config'] == source['config'] and not saved['config'].get('use_equiv_attn',False)
    model.load_state_dict(saved['model_state'],strict=True)
    model.to(args.device).eval()
    taps = {}
    def capture_nodes(module,inputs,output):
        taps['before_mask'] = pool_nodes(output,inputs[0].batch)
    def capture_readout(module,inputs):
        taps['after_mask'] = inputs[0].detach().clone()
    handles = [model.equi_update.register_forward_hook(capture_nodes),
               model.output_block.register_forward_pre_hook(capture_readout)]
    info = json.loads(manifest.read_text())['artifacts']['dielectric_total']
    assert sha(info['path']) == DATA_SHA
    records = {r['record_id']:r for r in map(json.loads,Path(info['path']).read_text().splitlines())}
    groups = info['available_point_groups']
    matrices,targets,predictions = {},{},[]
    try:
        for split,rows in splits.items():
            assert [r['sample_id'] for r in rows] == list(ids[split])
            assert set(ids[split]) == {k for k,r in records.items() if r['split']==split}
            targets[split] = torch.tensor([groups.index(records[k]['point_group']) for k in ids[split]])
            collected = {'before_mask':[],'after_mask':[]}
            for start in range(0,len(rows),source['config']['batch_size']):
                graph,mask,equality,_ = _collate(rows[start:start+source['config']['batch_size']],graphs.Data,Batch,args.device)
                taps.clear()
                with torch.enable_grad():
                    output = model(graph,mask,equality)
                assert taps['before_mask'].shape == (len(mask),32)
                torch.testing.assert_close(taps['after_mask'],torch.bmm(mask,taps['before_mask'].unsqueeze(-1)).squeeze(-1))
                for key in collected:
                    collected[key].append(taps[key].cpu())
                if split=='test':
                    predictions.append((.5*(output+output.transpose(-1,-2))).detach().cpu())
                del output,graph
            matrices[split] = {k:torch.cat(v) for k,v in collected.items()}
            print('extracted',split,len(rows),flush=True)
    finally:
        for handle in handles:
            handle.remove()
    reference = [json.loads(line) for line in Path(source['predictions']).read_text().splitlines()]
    assert [r['sample_id'] for r in reference] == list(ids['test'])
    actual = torch.cat(predictions)
    expected = torch.tensor([r['prediction'] for r in reference],dtype=actual.dtype)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
    reports = {}
    for key in ('before_mask','after_mask','shuffled_before_mask'):
        reports[key] = fit_probe({s:v[key.removeprefix('shuffled_')] for s,v in matrices.items()},targets,7,args.device,key.startswith('shuffled_'))
        print(key,reports[key]['test'],flush=True)
    assert sha(checkpoint) == CHECKPOINT_SHA
    report = dict(status='passed',source_run=498,checkpoint_epoch=196,checkpoint_sha256=CHECKPOINT_SHA,
        source_summary_sha256=SOURCE_SHA,dataset_sha256=DATA_SHA,groups=groups,
        split_counts={s:len(v) for s,v in ids.items()},test_ids=ids['test'],test_labels=targets['test'].tolist(),
        model_optimizer_steps=0,probes=reports,max_prediction_difference=float((actual-expected).abs().max()),
        seconds=time.monotonic()-started)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(features=matrices,ids=ids,labels=targets,checkpoint_sha256=CHECKPOINT_SHA),args.output/'features.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    torch.set_num_threads(4)
    run(parser.parse_args())
