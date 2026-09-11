"""Validate final dielectric/elastic symmetry for all 32 point groups.

The structures are reproducible synthetic Si/Ge crystals: two general-position
orbits in the representative symmorphic space group stored in the checked-in
point-group database.  They are test fixtures, not claims of stable compounds.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support.*")
warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated.*")

import numpy as np
import spglib
import torch
from e3nn.io import CartesianTensor

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from pg_tensor_model import CrystalInput, ModelConfig, PointGroupTensorModel  # noqa: E402
from pg_tensor_model.symmetry import (  # noqa: E402
    TARGET_IRREPS,
    PointGroupDAG,
    infer_point_group,
    invariant_basis,
)


SEEDS = (np.array([0.137, 0.271, 0.389]), np.array([0.193, 0.347, 0.421]))
SPECIES = (14, 32)
PAIR_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
PAIR_INDICES = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _unique_periodic(points: np.ndarray, tolerance: float = 1.0e-8) -> np.ndarray:
    unique: list[np.ndarray] = []
    for point in points % 1.0:
        if not any(
            np.max(np.abs((point - other + 0.5) % 1.0 - 0.5)) < tolerance
            for other in unique
        ):
            unique.append(point)
    return np.asarray(unique)


def representative_crystal(entry: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Generate two general Wyckoff orbits with exactly the requested PG."""
    symmetry = spglib.get_symmetry_from_database(entry["representative_hall_number"])
    rotations = np.asarray(symmetry["rotations"])
    translations = np.asarray(symmetry["translations"])

    # With column fractional coordinates, R.T G R = G.  If A.T A = G and the
    # spglib cell stores A.T, the corresponding Cartesian operations are orthogonal.
    metric = sum(rotation.T @ rotation for rotation in rotations)
    cell = 2.0 * np.linalg.cholesky(metric)

    fractional_orbits = []
    numbers = []
    for seed, atomic_number in zip(SEEDS, SPECIES):
        orbit = np.einsum("nij,j->ni", rotations, seed) + translations
        orbit = _unique_periodic(orbit)
        fractional_orbits.append(orbit)
        numbers.extend([atomic_number] * len(orbit))
    fractional = np.concatenate(fractional_orbits)

    positions = torch.tensor(fractional @ cell, dtype=torch.get_default_dtype())
    cell_tensor = torch.tensor(cell, dtype=torch.get_default_dtype())
    numbers_tensor = torch.tensor(numbers, dtype=torch.long)
    metadata = {
        "rotations": rotations,
        "translations": translations,
        "fractional": fractional,
        "first_orbit_size": len(fractional_orbits[0]),
    }
    return positions, cell_tensor, numbers_tensor, metadata


def symmetry_complete_edges(
    cell: torch.Tensor, numbers: torch.Tensor, metadata: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orbit a Si->Ge seed edge (and its reverse) under the full space group."""
    fractional = metadata["fractional"]
    rotations = metadata["rotations"]
    translations = metadata["translations"]
    first_ge = metadata["first_orbit_size"]
    edges: dict[tuple[int, int, tuple[float, float, float]], np.ndarray] = {}

    def mapped_index(raw: np.ndarray, atomic_number: int) -> int:
        wrapped = raw % 1.0
        candidates = np.flatnonzero(numbers.numpy() == atomic_number)
        deltas = (fractional[candidates] - wrapped + 0.5) % 1.0 - 0.5
        return int(candidates[np.linalg.norm(deltas, axis=1).argmin()])

    for source, target in ((0, first_ge), (first_ge, 0)):
        delta = fractional[target] - fractional[source]
        for rotation, translation in zip(rotations, translations):
            raw_source = rotation @ fractional[source] + translation
            raw_target = rotation @ fractional[target] + translation
            mapped_source = mapped_index(raw_source, int(numbers[source]))
            mapped_target = mapped_index(raw_target, int(numbers[target]))
            vector_fractional = rotation @ delta
            vector_cartesian = vector_fractional @ cell.numpy()
            key = (
                mapped_source,
                mapped_target,
                tuple(np.round(vector_cartesian, decimals=10)),
            )
            edges[key] = vector_cartesian

    edge_index = torch.tensor(
        [[key[0], key[1]] for key in edges], dtype=torch.long
    ).T
    edge_vectors = torch.tensor(
        np.stack(list(edges.values())), dtype=torch.get_default_dtype()
    )
    return edge_index, edge_vectors


def scalar_node_features(numbers: torch.Tensor, dimension: int) -> torch.Tensor:
    features = torch.zeros((len(numbers), dimension), dtype=torch.get_default_dtype())
    z = numbers.to(features.dtype) / 100.0
    for channel in range(8):
        features[:, channel] = z ** (channel + 1)
    return features


def _compact_components(task: str, tensor: torch.Tensor) -> tuple[list[str], torch.Tensor]:
    if task == "dielectric":
        labels = list(PAIR_LABELS)
        values = torch.stack([tensor[i, j] for i, j in PAIR_INDICES])
        return labels, values
    labels = []
    values = []
    for left, (i, j) in enumerate(PAIR_INDICES):
        for right in range(left, len(PAIR_INDICES)):
            k, ell = PAIR_INDICES[right]
            labels.append(f"C{left + 1}{right + 1}")
            values.append(tensor[i, j, k, ell])
    return labels, torch.stack(values)


def _forced_zero_labels(task: str, rotations: torch.Tensor) -> tuple[list[str], torch.Tensor]:
    irreps = TARGET_IRREPS[task]
    basis = invariant_basis(irreps, rotations)
    cartesian = CartesianTensor("ij=ji" if task == "dielectric" else "ijkl=ijlk=jikl=klij")
    basis_tensors = cartesian.to_cartesian(basis.T)
    labels, compact = zip(
        *[_compact_components(task, tensor) for tensor in basis_tensors]
    ) if basis_tensors.shape[0] else ([], [])
    if basis_tensors.shape[0]:
        # Every call has the same labels; stack coefficients by invariant basis vector.
        component_labels = labels[0]
        component_matrix = torch.stack(compact)
        forced = component_matrix.abs().amax(dim=0) < 1.0e-7
    else:
        component_labels, forced = [], torch.empty(0, dtype=torch.bool)
    return [label for label, is_zero in zip(component_labels, forced) if is_zero], forced


def _invariance_error(coefficients: torch.Tensor, rotations: torch.Tensor, task: str) -> float:
    errors = []
    scale = coefficients.norm().clamp_min(1.0e-12)
    for rotation in rotations:
        transformed = coefficients @ TARGET_IRREPS[task].D_from_matrix(rotation).T
        errors.append((transformed - coefficients).norm() / scale)
    return float(torch.stack(errors).amax())


def build_report() -> dict:
    torch.manual_seed(11)
    config = ModelConfig()
    model = PointGroupTensorModel(config).eval()
    dag = PointGroupDAG(config.dag_path)
    database = json.loads(Path(config.dag_path).read_text(encoding="utf-8"))
    entries = {
        value["hm_symbol"].replace(" ", ""): value
        for value in database["point_groups"].values()
    }
    rows = []

    with torch.no_grad():
        for symbol in dag.symbols:
            entry = entries[symbol]
            positions, cell, numbers, metadata = representative_crystal(entry)
            edge_index, edge_vectors = symmetry_complete_edges(cell, numbers, metadata)
            detected = infer_point_group(positions, cell, numbers, config.symprec)
            active = dag.ancestors(symbol)
            crystal = CrystalInput(
                positions=positions,
                cell=cell,
                atomic_numbers=numbers,
                node_features=scalar_node_features(numbers, model.hidden_irreps.dim),
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                current_point_group=symbol,
                parent_residuals={key: (0.0 if key == symbol else 1.0) for key in active},
            )
            task_results = {}
            for task in ("dielectric", "elastic"):
                output = model(crystal, task)
                labels, compact = _compact_components(task, output.tensor[0])
                zero_labels, forced_mask = _forced_zero_labels(
                    task, dag.records[symbol].cartesian_rotations
                )
                if forced_mask.any():
                    zero_leakage = float(compact[forced_mask].abs().amax())
                else:
                    zero_leakage = 0.0
                scale = float(compact.abs().amax().clamp_min(1.0))
                invariance_error = _invariance_error(
                    output.irrep_coefficients,
                    dag.records[symbol].cartesian_rotations.to(output.irrep_coefficients),
                    task,
                )
                task_results[task] = {
                    "forced_zero_components": zero_labels,
                    "max_forced_zero_abs": zero_leakage,
                    "relative_zero_leakage": zero_leakage / scale,
                    "max_relative_invariance_error": invariance_error,
                    "passed": zero_leakage / scale < 1.0e-6 and invariance_error < 1.0e-5,
                }
            rows.append(
                {
                    "number": entry["number"],
                    "point_group": symbol,
                    "schoenflies": entry["schoenflies"],
                    "example": (
                        f"synthetic Si{metadata['first_orbit_size']}"
                        f"Ge{len(numbers) - metadata['first_orbit_size']} "
                        f"({entry['representative_space_group_symbol']})"
                    ),
                    "space_group": entry["representative_space_group_symbol"],
                    "num_atoms": len(numbers),
                    "detected_point_group": detected,
                    **task_results,
                }
            )
    return {
        "method": "two-general-orbit synthetic crystals; symmetry-complete edge orbit; scalar O(3) input features",
        "tolerances": {"relative_zero_leakage": 1.0e-6, "relative_invariance": 1.0e-5},
        "point_groups": rows,
    }


def write_report(report: dict) -> None:
    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "point_group_output_symmetry.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Final-output symmetry validation for all 32 point groups",
        "",
        "Each example is a reproducible synthetic Si/Ge crystal built from two general-position ",
        "orbits of the listed representative space group. These are numerical fixtures, not claims ",
        "of thermodynamic stability. `zeros` lists conventional Cartesian/Voigt components forced ",
        "to zero by the representative point-group embedding. `none` means symmetry forces no ",
        "individual component to zero (it may still impose equalities between components).",
        "Elastic Voigt indices use 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy.",
        "",
        "| # | PG | fixture crystal | atoms | spglib | dielectric zeros | dielectric | elastic zeros | elastic |",
        "|---:|:---:|:---:|---:|:---:|:---|:---:|:---|:---:|",
    ]
    for row in report["point_groups"]:
        dielectric = row["dielectric"]
        elastic = row["elastic"]
        dz = ", ".join(dielectric["forced_zero_components"]) or "none"
        ez = ", ".join(elastic["forced_zero_components"]) or "none"
        lines.append(
            f"| {row['number']} | {row['point_group']} | {row['example']} | "
            f"{row['num_atoms']} | {row['detected_point_group']} | {dz} | "
            f"{'PASS' if dielectric['passed'] else 'FAIL'} "
            f"({dielectric['relative_zero_leakage']:.2e}) | {ez} | "
            f"{'PASS' if elastic['passed'] else 'FAIL'} "
            f"({elastic['relative_zero_leakage']:.2e}) |"
        )
    lines.extend(
        [
            "",
            "The number in parentheses is maximum forced-zero leakage divided by "
            "`max(1, max_abs_output)`. Full group-invariance residuals are retained in the JSON report.",
        ]
    )
    (output_dir / "point_group_output_symmetry.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    result = build_report()
    write_report(result)
    failures = [
        (row["point_group"], task)
        for row in result["point_groups"]
        for task in ("dielectric", "elastic")
        if not row[task]["passed"]
    ]
    print(f"validated {len(result['point_groups'])} point groups; failures={failures}")
