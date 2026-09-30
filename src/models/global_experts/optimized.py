"""Algorithm-preserving grouped gates and bounded geometric representation cache."""
from collections import OrderedDict

import torch
from torch import nn

from ...experts.modules import FullPointGroupExpert


class GroupedFiniteGroupBlock(nn.Module):
    def __init__(self, reference):
        super().__init__()
        for name in ("representation", "invariant_projector"):
            self.register_buffer(name, getattr(reference, name))
        for name in ("weight", "bias", "gate_gain", "gate_bias"):
            self.register_parameter(name, getattr(reference, name))
        self.copy_slices = reference.copy_slices
        self.trivial_copies = reference.trivial_copies
        groups = {}
        for index, (part, trivial) in enumerate(zip(self.copy_slices, self.trivial_copies)):
            groups.setdefault((part.stop - part.start, trivial), []).append(index)
        self.groups = tuple(groups)
        order = []
        for k, ((width, trivial), copies) in enumerate(groups.items()):
            coordinates = [j for i in copies for j in range(self.copy_slices[i].start, self.copy_slices[i].stop)]
            order.extend(coordinates)
            self.register_buffer(f"coordinates_{k}", torch.tensor(coordinates), persistent=False)
            self.register_buffer(f"copies_{k}", torch.tensor(copies), persistent=False)
        self.register_buffer("restore", torch.argsort(torch.tensor(order)), persistent=False)

    def forward(self, features):
        representation = self.representation.to(features)
        weight = torch.einsum("gij,jk,glk->il", representation, self.weight, representation) / len(representation)
        hidden = features @ weight + self.bias @ self.invariant_projector.to(features)
        outputs = []
        for k, (width, trivial) in enumerate(self.groups):
            copies = getattr(self, f"copies_{k}")
            values = hidden.index_select(-1, getattr(self, f"coordinates_{k}"))
            values = values.reshape(*hidden.shape[:-1], len(copies), width)
            if trivial:
                gated = torch.nn.functional.silu(values)
            else:
                norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
                gain = self.gate_gain.index_select(0, copies).unsqueeze(-1)
                bias = self.gate_bias.index_select(0, copies).unsqueeze(-1)
                gated = values * torch.sigmoid(gain * norm + bias)
            outputs.append(gated.flatten(-2))
        return features + torch.cat(outputs, dim=-1).index_select(-1, self.restore)


class GroupedFullPointGroupExpert(FullPointGroupExpert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([
            block if isinstance(block, nn.Identity) else GroupedFiniteGroupBlock(block)
            for block in self.blocks
        ])


class FrameRepresentationCache:
    def __init__(self, capacity=8192):
        self.capacity = capacity
        self.values = OrderedDict()

    def clear(self):
        self.values.clear()

    def get(self, irreps, frame, reference, *, enabled=True):
        frame = frame.detach().to(device="cpu", dtype=reference.dtype).contiguous()
        # Value-based keys also invalidate in-place metadata edits and rotation augmentation.
        key = (str(irreps), reference.dtype, str(reference.device), frame.shape, frame.numpy().tobytes())
        if enabled and key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        if frame.shape != (3, 3) or not torch.allclose(
            frame @ frame.T, torch.eye(3, dtype=frame.dtype), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("standard frame must be orthogonal")
        result = irreps.D_from_matrix(frame).to(reference)
        if enabled:
            self.values[key] = result
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return result
