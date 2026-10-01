"""Slurm preparation stages; persist small invariant embeddings, not full latent tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from ...backbones import BackboneResourceRegistry
from ...baselines.gmtnet.runner import (
    _load_official_modules, _plain_graph, _prepare_row, dpa4_invariant_node_embedding,
)
from ...cli.reporting import execution_metadata
from ...irreps import IrrepLayout, IrrepTerm
from ...models import build_backbone_adapter
from ...training import extract_frozen_examples
from .data import dataset_from_records, official_voigt
from .protocol import FRACTIONS, load_records, nested_ids, write_json


def synchronize(device="cuda"):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def official_modules(root, task):
    root = Path(root) / ("GMTNet_elast" if task == "elastic" else "")
    modules = _load_official_modules(root)
    if any(Path(module.__file__).resolve().parent != root.resolve() for module in modules):
        raise RuntimeError("run each task in a fresh process to isolate official module names")
    return modules


def prepare_graph(sample, official_data, official_graphs):
    from pymatgen.io.jarvis import JarvisAtomsAdaptor
    adaptor = JarvisAtomsAdaptor()
    if sample.unit.target == "dielectric":
        return _prepare_row(sample, official_data, official_graphs, adaptor)
    import numpy as np
    from pymatgen.core import Structure
    structure = Structure(sample.lattice.numpy(), sample.atomic_numbers.tolist(),
                          sample.fractional_positions.numpy())
    symmetry = official_data.get_symmetry_dataset(structure, symprec=1e-5)
    rotations = official_data.rm_duplicates(np.asarray(symmetry["rotations"]))
    lattice = structure.lattice.matrix.T
    cartesian = lattice @ rotations @ np.linalg.inv(lattice)
    if not official_data.is_group(cartesian):
        raise ValueError("invalid elastic structure rotation group")
    matrices = official_data.irreps_output.D_from_matrix(torch.tensor(cartesian, dtype=torch.float32))
    summed = matrices.sum(0)
    marker = torch.arange(73, dtype=torch.float32) + 10
    marker[16:] *= 100
    marker = summed @ marker
    indices = [0, 1, *range(16, 26), *range(64, 73)]
    ideal = official_voigt(official_data.converter.to_cartesian(marker[indices]))
    mask = summed / len(matrices)
    mask *= mask > 1e-5
    graph = official_graphs.atoms2graphs(
        adaptor.get_atoms(structure), cutoff=4.0, max_neighbors=16, reduce=False,
        equivalent_atoms=symmetry["equivalent_atoms"], use_canonize=True)
    return {"sample_id": sample.sample_id, "graph": _plain_graph(graph),
            "feature_mask": mask.cpu(), "equality": official_data.find_almost_equal_entries(ideal).cpu(),
            "target": official_voigt(sample.target_cartesian).float().cpu()}


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def prepare(args):
    rows, provenance = load_records(args.task, group_scope=args.group_scope)
    if args.smoke or args.inference:
        ids = nested_ids(rows, 100, smoke=args.smoke)
        selected = (set(sid for fraction in FRACTIONS
                        for values in nested_ids(rows, fraction, smoke=True).values() for sid in values)
                    if args.smoke else set(ids["test"][:32]))
        # A dataset contract needs all splits; inference filters after construction.
        if args.smoke:
            rows = [r for r in rows if r["record_id"] in selected]
    dataset = dataset_from_records(rows, provenance)
    samples = tuple(dataset)
    if args.inference:
        samples = tuple(dataset.by_id(sid) for sid in dataset.split_manifest.test[:32])
    output = args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preparation artifact: {output}")
    start = time.perf_counter()
    by_id = {}
    times = {}
    initialization_start = time.perf_counter()
    if args.mode == "features":
        resource = BackboneResourceRegistry()["dpa4"]
        adapter = build_backbone_adapter("dpa4", IrrepLayout((IrrepTerm(1, 0, "e", "unused"),)),
                                         device=args.device)
        synchronize(args.device)
    else:
        _, graphs, data = official_modules(args.official_root, args.task)
    initialization_seconds = time.perf_counter() - initialization_start
    for index, sample in enumerate(samples):
        synchronize(args.device)
        sample_start = time.perf_counter()
        if args.mode == "features":
            layout, examples = extract_frozen_examples(adapter, (sample,),
                                                       cutoff=resource.cutoff_angstrom, device=args.device)
            example = examples[0]
            if not torch.equal(example.graph.atomic_numbers, sample.atomic_numbers):
                raise ValueError("feature extraction changed atom ordering")
            value = {"embedding": dpa4_invariant_node_embedding(example.features, layout).float(),
                     "atomic_numbers": example.graph.atomic_numbers}
        else:
            value = prepare_graph(sample, data, graphs)
        synchronize(args.device)
        times[sample.sample_id] = time.perf_counter() - sample_start
        by_id[sample.sample_id] = value
        if index % 100 == 0:
            print(json.dumps({"stage": args.mode, "task": args.task, "completed": index+1,
                              "total": len(samples)}), flush=True)
    payload = {"provenance": provenance, "by_id": by_id, "seconds_by_id": times,
               "initialization_seconds": initialization_seconds, "smoke": args.smoke,
               "inference": args.inference, "execution": json.loads(json.dumps(execution_metadata()))}
    if args.mode == "features":
        payload["checkpoint_sha256"] = resource.sha256
    save_start = time.perf_counter()
    save_tensor(output, payload)
    save_seconds = time.perf_counter() - save_start
    metadata = {key: value for key, value in payload.items() if key != "by_id"}
    metadata.update({"total_seconds": time.perf_counter()-start, "save_seconds": save_seconds,
                     "sample_count": len(samples), "path": str(output)})
    write_json(output.with_suffix(".json"), metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("dielectric", "elastic"), required=True)
    parser.add_argument("--mode", choices=("features", "graphs"), required=True)
    parser.add_argument("--group-scope", choices=("all", "seven"), default="all")
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inference", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
