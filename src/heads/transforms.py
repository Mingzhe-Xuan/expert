from __future__ import annotations

import torch

# Importing the registry installs the narrow PyTorch/e3nn safe-global compatibility
# declaration before CartesianTensor loads e3nn's packaged Wigner constants.
from ..symmetry.registry import PointGroup
from .contracts import TARGET_LAYOUTS
from e3nn.io import CartesianTensor


_CARTESIAN_FORMULAS = {
    "dielectric": "ij=ji",
    "elastic": "ijkl=ijlk=jikl=klij",
    "bec": "ij",
}
_CARTESIAN_TRANSFORMS = {
    task: CartesianTensor(formula) for task, formula in _CARTESIAN_FORMULAS.items()
}
_ORTHONORMAL_BASES: dict[str, torch.Tensor] = {}


def _orthonormal_change_of_basis(task: str, reference: torch.Tensor) -> torch.Tensor:
    """Return a closest-orthonormal float64 repair of e3nn's generated basis.

    e3nn 0.5 generates the symbolic basis through default-float intermediates. Merely
    casting its result to float64 leaves round-trip errors around 1e-7. A polar factor
    fixes this once, without changing the represented Cartesian tensor subspace.
    """

    if task not in _ORTHONORMAL_BASES:
        transform = _target_transform(task)
        seed = torch.zeros(1, transform.dim, dtype=torch.float64)
        raw = transform.reduced_tensor_products(seed).change_of_basis
        flat = raw.to(dtype=torch.float64, device="cpu").flatten(1)
        left, _, right = torch.linalg.svd(flat, full_matrices=False)
        _ORTHONORMAL_BASES[task] = (left @ right).contiguous()
    return _ORTHONORMAL_BASES[task].to(device=reference.device, dtype=reference.dtype)


def cartesian_to_irreps(tensor: torch.Tensor, task: str) -> torch.Tensor:
    """Convert a task Cartesian tensor to the frozen real-irrep copy order."""

    transform = _target_transform(task)
    basis = _orthonormal_change_of_basis(task, tensor)
    coefficients = tensor.flatten(-len(transform.indices)) @ basis.T
    if coefficients.shape[-1] != TARGET_LAYOUTS[task].dimension:
        raise RuntimeError("CartesianTensor order disagrees with the frozen target layout")
    return coefficients


def irreps_to_cartesian(coefficients: torch.Tensor, task: str) -> torch.Tensor:
    """Convert frozen-order real-irrep coefficients to Cartesian components."""

    transform = _target_transform(task)
    if coefficients.ndim < 1 or coefficients.shape[-1] != TARGET_LAYOUTS[task].dimension:
        raise ValueError("coefficient width does not match the target layout")
    basis = _orthonormal_change_of_basis(task, coefficients)
    flat = coefficients @ basis
    return flat.reshape(*coefficients.shape[:-1], *((3,) * len(transform.indices)))


def _target_transform(task: str) -> CartesianTensor:
    try:
        return _CARTESIAN_TRANSFORMS[task]
    except KeyError as error:
        raise ValueError(f"unsupported target task {task!r}") from error


def rotate_cartesian(
    tensor: torch.Tensor, rotation: torch.Tensor, task: str
) -> torch.Tensor:
    """Apply an active Cartesian O(3) transform to one task tensor batch."""

    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    rotation = rotation.to(tensor)
    if task in {"dielectric", "bec"}:
        if tensor.shape[-2:] != (3, 3):
            raise ValueError(f"{task} tensor must end in [3, 3]")
        return torch.einsum("ia,...ab,jb->...ij", rotation, tensor, rotation)
    if task == "elastic":
        if tensor.shape[-4:] != (3, 3, 3, 3):
            raise ValueError("elastic tensor must end in [3, 3, 3, 3]")
        return torch.einsum(
            "ia,jb,kc,ld,...abcd->...ijkl",
            rotation,
            rotation,
            rotation,
            rotation,
            tensor,
        )
    raise ValueError(f"unsupported target task {task!r}")


def target_representation(
    rotation: torch.Tensor, task: str, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return the frozen target coefficient representation for one O(3) matrix."""

    layout = TARGET_LAYOUTS.get(task)
    if layout is None:
        raise ValueError(f"unsupported target task {task!r}")
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    if dtype is not None:
        rotation = rotation.to(dtype=dtype)
    reference = rotation.new_empty(())
    basis = _orthonormal_change_of_basis(task, reference)
    cartesian_basis = basis.reshape(layout.dimension, *((3,) * len(_target_transform(task).indices)))
    rotated_basis = rotate_cartesian(cartesian_basis, rotation, task)
    # Rows contain images of coefficient basis vectors; transpose to D such that
    # row-vector coefficients transform as c' = c @ D.T.
    return (rotated_basis.flatten(1) @ basis.T).T


def project_to_point_group(
    coefficients: torch.Tensor, task: str, group: PointGroup
) -> torch.Tensor:
    """Project global target coefficients into a point group's fixed subspace."""

    if task == "bec":
        raise ValueError("BEC raw predictions must not use the global fixed-space projector")
    layout = TARGET_LAYOUTS.get(task)
    if layout is None:
        raise ValueError(f"unsupported target task {task!r}")
    if coefficients.shape[-1] != layout.dimension:
        raise ValueError("coefficient width does not match the target layout")
    # Fixed-space rank decisions are convention data, not model precision.  In
    # float32, the 1e-9 deterministic range threshold can promote round-off
    # residue to forbidden basis vectors (notably for cubic groups).
    basis = group.invariant_basis(layout, dtype=torch.float64).to(coefficients)
    return (coefficients @ basis) @ basis.T


def project_to_symmetry_operations(
    coefficients: torch.Tensor,
    task: str,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Project a global target using the material's detected Cartesian operations."""

    if task == "bec":
        raise ValueError("BEC raw predictions must not use the global fixed-space projector")
    layout = TARGET_LAYOUTS.get(task)
    if layout is None:
        raise ValueError(f"unsupported target task {task!r}")
    if coefficients.ndim < 1 or coefficients.shape[-1] != layout.dimension:
        raise ValueError("coefficient width does not match the target layout")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or rotations.shape[0] < 1:
        raise ValueError("rotations must be non-empty with shape [num_operations, 3, 3]")
    if rotations.device != coefficients.device:
        raise ValueError("coefficients and rotations must share a device")
    representations = torch.stack(
        [target_representation(rotation, task, dtype=coefficients.dtype) for rotation in rotations]
    )
    projector = representations.mean(dim=0)
    projector = 0.5 * (projector + projector.T)
    return coefficients @ projector.T


def apply_bec_asr(bec: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    """Enforce the acoustic sum rule independently in every crystal."""

    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        raise ValueError("BEC must have shape [num_atoms, 3, 3]")
    if node_batch.shape != (bec.shape[0],) or node_batch.dtype != torch.long:
        raise TypeError("node_batch must be torch.long with shape [num_atoms]")
    if node_batch.device != bec.device:
        raise ValueError("BEC and node_batch must share a device")
    if not node_batch.numel():
        return bec.clone()
    if int(node_batch.min()) < 0:
        raise ValueError("node_batch indices must be non-negative")
    graph_count = int(node_batch.max()) + 1
    sums = bec.new_zeros((graph_count, 3, 3))
    sums.index_add_(0, node_batch, bec)
    counts = bec.new_zeros(graph_count)
    counts.index_add_(0, node_batch, torch.ones_like(node_batch, dtype=bec.dtype))
    if bool((counts == 0).any()):
        raise ValueError("node_batch graph indices must be contiguous")
    return bec - (sums / counts[:, None, None])[node_batch]


def project_bec_joint_symmetry(
    bec: torch.Tensor,
    rotations: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    """Optional diagnostic Reynolds control over sites and rank-2 fibers.

    This function is deliberately separate from the raw BEC head. ``permutations[g, i]``
    is the image site ``pi_g(i)`` and the projected result obeys
    ``Z[pi_g(i)] = R_g Z[i] R_g.T`` when the supplied operations form a valid action.
    """

    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        raise ValueError("BEC must have shape [num_atoms, 3, 3]")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape [num_operations, 3, 3]")
    if permutations.shape != (rotations.shape[0], bec.shape[0]):
        raise ValueError("permutations must have shape [num_operations, num_atoms]")
    if permutations.dtype != torch.long:
        raise TypeError("permutations must use torch.long")
    if rotations.device != bec.device or permutations.device != bec.device:
        raise ValueError("BEC, rotations and permutations must share a device")
    expected = torch.arange(bec.shape[0], device=bec.device)
    for permutation in permutations:
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("each symmetry site mapping must be a permutation")
    projected = torch.zeros_like(bec)
    for rotation, permutation in zip(rotations, permutations):
        transformed = rotate_cartesian(bec, rotation, "bec")
        projected.index_add_(0, permutation, transformed)
    return projected / rotations.shape[0]


def decanonicalize_cartesian(
    tensor: torch.Tensor,
    task: str,
    canonical_to_input: torch.Tensor,
    canonical_to_input_sites: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore input frame and, for node BEC, the original site order."""

    restored = rotate_cartesian(tensor, canonical_to_input, task)
    if task == "bec":
        if canonical_to_input_sites is None:
            raise ValueError("BEC de-canonicalization requires a site-order mapping")
        if canonical_to_input_sites.shape != (tensor.shape[0],):
            raise ValueError("site-order mapping must have shape [num_atoms]")
        expected = torch.arange(tensor.shape[0], device=tensor.device)
        if canonical_to_input_sites.dtype != torch.long or not torch.equal(
            torch.sort(canonical_to_input_sites).values, expected
        ):
            raise ValueError("site-order mapping must be a permutation")
        output = torch.empty_like(restored)
        output[canonical_to_input_sites] = restored
        return output
    if canonical_to_input_sites is not None:
        raise ValueError("global targets do not accept a site-order mapping")
    return restored
