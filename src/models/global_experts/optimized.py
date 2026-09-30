"""Algorithm-preserving grouped gates and bounded geometric representation cache."""
from collections import OrderedDict

import torch
from ...experts.optimized import GroupedFiniteGroupBlock, GroupedFullPointGroupExpert




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
