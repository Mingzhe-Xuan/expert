"""Four-way matched-width softmax probing with frozen-ID aligned feature caches."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from .gmtnet_pg import run as extract, sha
from .pg_linear import DATA_SHA
from .logistic import fit_logistic

SOURCES = {
    'dpa_pca32':('results/pca-pg-probe/536/pca_features.pt','a35ae81c7ced910e7043f2c5c7027b2ce039c9d13005f971492ef4403a73023f',None),
    'dpa_gmtnet':('results/gmtnet-pg-probe/537/features.pt','7d210cecd2b54ee712618b08463415581c606b8f8da6903f3ce54de0a87ba933','before_mask'),
    'global_pg':('results/fused-pg-probe/535/features.pt','64cdc5744738fe4ca7514e0c06eeff4148bbff0c61ef0602aa4c026e3af109a8','fused_before_mask'),
}


def run(args):
    started = time.monotonic()
    base_dir = args.output/'gmtnet_extraction'
    baseline = extract(SimpleNamespace(official_root=args.official_root,output=base_dir,device=args.device,baseline=True,extract_only=True))
    sources = {**SOURCES,'gmtnet':(str(base_dir/'features.pt'),sha(base_dir/'features.pt'),'before_mask')}
    manifest = json.loads(Path('data/manifests/curated_tensors_reduced_gt_5pct.json').read_text())['artifacts']['dielectric_total']
    assert sha(manifest['path']) == DATA_SHA
    records = [json.loads(line) for line in Path(manifest['path']).read_text().splitlines()]
    groups = manifest['available_point_groups']
    reference = {s:{r['record_id']:groups.index(r['point_group']) for r in records if r['split']==s} for s in ('train','validation','test')}
    ids = {s:sorted(v) for s,v in reference.items()}
    labels = {s:torch.tensor([reference[s][key] for key in ids[s]]) for s in ids}
    assert {s:len(v) for s,v in ids.items()} == dict(train=5001,validation=637,test=677)
    reports,states,provenance = {},{},{}
    for name,(path,digest,tap) in sources.items():
        assert sha(path) == digest
        cache = torch.load(path,map_location='cpu',weights_only=True)
        matrices = {}
        for split in ids:
            old_ids = list(cache['ids'][split])
            assert len(old_ids) == len(set(old_ids)) == len(ids[split])
            assert dict(zip(old_ids,cache['labels'][split].tolist())) == reference[split]
            index = {key:i for i,key in enumerate(old_ids)}
            value = cache['features'][split] if tap is None else cache['features'][split][tap]
            matrices[split] = value[[index[key] for key in ids[split]]]
            assert matrices[split].shape == (len(ids[split]),32)
        for shuffle in (False,True):
            key = name+'_shuffled' if shuffle else name
            reports[key],states[key] = fit_logistic(matrices,labels,len(groups),shuffle)
            print(key,reports[key]['alpha'],reports[key]['test'],flush=True)
        provenance[name] = dict(path=path,sha256=digest,tap=tap or 'train_only_PCA32')
    report = dict(status='passed',classifier='multinomial_softmax_cross_entropy_L2',
        objective='mean cross entropy + 0.5 * alpha * squared weight norm; intercept unpenalized',
        class_weighting='none',selection='validation_macro_f1_then_accuracy_then_larger_alpha',
        standardization='train_only',feature_tap='before_explicit_symmetry_mask',
        model_optimizer_steps=0,dataset_sha256=DATA_SHA,groups=groups,
        split_counts={s:len(v) for s,v in ids.items()},test_ids=ids['test'],test_labels=labels['test'].tolist(),
        checkpoint_epochs=dict(gmtnet=93,dpa_gmtnet=196,global_pg=196),
        gmtnet_historical_pre100_exception=True,gmtnet_extraction=baseline,
        provenance=provenance,probes=reports,seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'classifiers.json').write_text(json.dumps(states,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--official-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    torch.set_num_threads(4)
    run(parser.parse_args())
