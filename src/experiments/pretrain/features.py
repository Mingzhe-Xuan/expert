"""Native-neighbor extraction with the existing canonical frame and parity formula."""
from __future__ import annotations

import torch

from ...baselines.gmtnet.runner import dpa4_invariant_node_embedding
from ...graphs import PeriodicGraph
from ...symmetry import canonicalize_structure


def geometry_input(positions, cell, numbers, cutoff):
    """Geometry carrier for an extractor that always constructs its own neighbors."""
    return PeriodicGraph(
        positions=positions, cell=cell.unsqueeze(0), atomic_numbers=numbers,
        node_batch=torch.zeros(len(numbers), dtype=torch.long, device=positions.device),
        edge_index=torch.empty((2,0), dtype=torch.long, device=positions.device),
        cell_shifts=torch.empty((0,3), dtype=torch.long, device=positions.device),
        edge_vectors=positions.new_empty((0,3)), edge_distances=positions.new_empty((0,)),
        cutoff=cutoff, boundary_convention="geometry_only_native_neighbor_input")


def extract_embedding(adapter, sample, *, device):
    canonical = canonicalize_structure(sample.cartesian_positions.to(device=device,dtype=torch.float32),
                                       sample.lattice.to(device=device,dtype=torch.float32),
                                       sample.atomic_numbers.to(device))
    graph = geometry_input(canonical.canonical_positions, canonical.canonical_cell,
                           canonical.atomic_numbers, adapter.resource.cutoff_angstrom)
    fractional = torch.linalg.solve(graph.cell[0].T, graph.positions.T).T
    inverted = geometry_input(torch.remainder(-fractional,1.0) @ graph.cell[0], graph.cell[0],
                              graph.atomic_numbers, graph.cutoff)
    with torch.no_grad():
        direct, reverse = adapter.extractor(graph), adapter.extractor(inverted)
    if direct.node_layout != adapter.parity.source_layout or reverse.node_layout != direct.node_layout:
        raise ValueError("native source layout mismatch")
    if not torch.equal(direct.node_batch, graph.node_batch) or not torch.equal(reverse.node_batch, graph.node_batch):
        raise ValueError("native extractor reordered atoms")
    blocks=[]
    offset=0
    for term in adapter.parity.source_layout.terms:
        stop=offset+term.dimension
        a,b=direct.node_features[:,offset:stop],reverse.node_features[:,offset:stop]
        blocks.extend((.5*(a+b),.5*(a-b)))
        offset=stop
    return dpa4_invariant_node_embedding(torch.cat(blocks,-1),adapter.parity.output_layout).float().cpu()
