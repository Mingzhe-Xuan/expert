from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from ..heads import TARGET_LAYOUTS, cartesian_to_irreps
from .contracts import SplitManifest, TrainingUnit, make_seeded_split


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATA_MANIFESTS = MappingProxyType(
    {
        ("jarvis_tensor", "dielectric"): DATA_ROOT / "manifests" / "jarvis_dielectric.json",
        ("jarvis_tensor", "elastic"): DATA_ROOT / "manifests" / "jarvis_elastic.json",
        ("dtnet", "dielectric"): DATA_ROOT / "manifests" / "dtnet_dielectric.json",
        ("matten", "elastic"): DATA_ROOT / "manifests" / "matten_elastic.json",
        ("jarvis_dfpt", "bec"): DATA_ROOT / "manifests" / "jarvis_dfpt_bec.json",
    }
)
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True, slots=True)
class TensorSample:
    sample_id: str
    unit: TrainingUnit
    lattice: torch.Tensor
    fractional_positions: torch.Tensor
    atomic_numbers: torch.Tensor
    target_cartesian: torch.Tensor
    target_coefficients: torch.Tensor
    target_unit: str
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.target_unit:
            raise ValueError("tensor samples require a stable ID and target unit")
        n_atoms = self.fractional_positions.shape[0]
        if self.lattice.shape != (3, 3):
            raise ValueError("sample lattice must have shape [3, 3]")
        if self.fractional_positions.shape != (n_atoms, 3) or n_atoms < 1:
            raise ValueError("fractional positions must have shape [num_atoms, 3]")
        if self.atomic_numbers.shape != (n_atoms,) or self.atomic_numbers.dtype != torch.long:
            raise TypeError("atomic numbers must be torch.long with one value per atom")
        expected_cartesian = (n_atoms, 3, 3) if self.unit.target == "bec" else (
            (3, 3) if self.unit.target == "dielectric" else (3, 3, 3, 3)
        )
        expected_coefficients = (
            (n_atoms, TARGET_LAYOUTS[self.unit.target].dimension)
            if self.unit.target == "bec"
            else (TARGET_LAYOUTS[self.unit.target].dimension,)
        )
        if self.target_cartesian.shape != expected_cartesian:
            raise ValueError("sample Cartesian target has the wrong scope or shape")
        if self.target_coefficients.shape != expected_coefficients:
            raise ValueError("sample coefficient target has the wrong scope or width")
        tensors = (
            self.lattice,
            self.fractional_positions,
            self.target_cartesian,
            self.target_coefficients,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("tensor samples must contain only finite values")
        if bool(((self.atomic_numbers < 1) | (self.atomic_numbers > 118)).any()):
            raise ValueError("atomic numbers must be in the range 1..118")
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))

    @property
    def cartesian_positions(self) -> torch.Tensor:
        return self.fractional_positions @ self.lattice

    def build_graph(self, cutoff: float):
        """Build a deterministic periodic graph without changing stored site order."""

        from ..graphs import build_periodic_graph

        return build_periodic_graph(
            self.cartesian_positions,
            self.lattice,
            self.atomic_numbers,
            cutoff,
        )


class IndependentTensorDataset(Sequence[TensorSample]):
    """One immutable dataset-by-property unit with an independent split manifest."""

    def __init__(
        self,
        unit: TrainingUnit,
        samples: Sequence[TensorSample],
        split_manifest: SplitManifest,
    ) -> None:
        if any(sample.unit != unit for sample in samples):
            raise ValueError("all samples must belong to the dataset training unit")
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("dataset sample IDs must be unique")
        allowed = set(split_manifest.train + split_manifest.validation + split_manifest.test)
        if not set(sample_ids) <= allowed:
            raise ValueError("dataset contains samples outside its split manifest")
        self.unit = unit
        self.samples = tuple(samples)
        self.split_manifest = split_manifest
        self._by_id = MappingProxyType(dict(zip(sample_ids, self.samples)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TensorSample:
        return self.samples[index]

    def by_id(self, sample_id: str) -> TensorSample:
        try:
            return self._by_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"sample {sample_id!r} was not loaded") from exc

    def five_structure_smoke(self) -> tuple[TensorSample, ...]:
        ids = (
            *self.split_manifest.train[:3],
            self.split_manifest.validation[0],
            self.split_manifest.test[0],
        )
        missing = [sample_id for sample_id in ids if sample_id not in self._by_id]
        if missing:
            raise ValueError(f"five-structure smoke samples were not loaded: {missing}")
        return tuple(self._by_id[sample_id] for sample_id in ids)


@dataclass(frozen=True, slots=True)
class StructureCandidate:
    """One equilibrium structure candidate before point-group fixture selection."""

    sample_id: str
    source_dataset: str
    source_manifest_sha256: str
    lattice: torch.Tensor
    fractional_positions: torch.Tensor
    atomic_numbers: torch.Tensor

    def __post_init__(self) -> None:
        node_count = self.fractional_positions.shape[0]
        if not self.sample_id or not self.source_dataset:
            raise ValueError("structure candidates require source and stable sample ID")
        if len(self.source_manifest_sha256) != 64:
            raise ValueError("source manifest SHA-256 is required")
        if self.lattice.shape != (3, 3) or abs(float(torch.linalg.det(self.lattice))) < 1e-12:
            raise ValueError("candidate lattice must be invertible with shape [3, 3]")
        if self.fractional_positions.shape != (node_count, 3) or node_count < 1:
            raise ValueError("candidate fractional positions must be non-empty [N, 3]")
        if self.atomic_numbers.shape != (node_count,) or self.atomic_numbers.dtype != torch.long:
            raise TypeError("candidate atomic numbers must be torch.long [N]")
        if not torch.isfinite(self.lattice).all() or not torch.isfinite(
            self.fractional_positions
        ).all():
            raise ValueError("candidate geometry must contain only finite values")
        if bool(((self.atomic_numbers < 1) | (self.atomic_numbers > 118)).any()):
            raise ValueError("candidate atomic numbers must be in the range 1..118")

    @property
    def cartesian_positions(self) -> torch.Tensor:
        return self.fractional_positions @ self.lattice


def voigt_stiffness_to_cartesian(voigt: torch.Tensor) -> torch.Tensor:
    """Expand a stiffness Voigt matrix without engineering-strain scale factors."""

    if voigt.shape != (6, 6):
        raise ValueError("elastic Voigt matrix must have shape [6, 6]")
    output = voigt.new_empty((3, 3, 3, 3))
    for a, (i, j) in enumerate(_VOIGT_PAIRS):
        for b, (k, ell) in enumerate(_VOIGT_PAIRS):
            value = voigt[a, b]
            output[i, j, k, ell] = value
            output[j, i, k, ell] = value
            output[i, j, ell, k] = value
            output[j, i, ell, k] = value
            output[k, ell, i, j] = value
            output[ell, k, i, j] = value
            output[k, ell, j, i] = value
            output[ell, k, j, i] = value
    return output


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _resource_path(manifest_path: Path, relative: str) -> Path:
    data_root = manifest_path.parent.parent.resolve()
    path = (data_root / relative).resolve()
    if path != data_root and data_root not in path.parents:
        raise ValueError("dataset resource path escapes the data root")
    return path


def _verify_resource(path: Path, *, size_bytes: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset resource: {path}")
    if path.stat().st_size != int(size_bytes):
        raise ValueError(f"dataset resource size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != str(sha256).lower():
        raise ValueError(f"dataset resource SHA-256 mismatch: {path.name}")


def _atomic_numbers(elements: Sequence[str]) -> torch.Tensor:
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise ImportError("real-data modules require ASE for element symbols") from exc
    try:
        values = [atomic_numbers[str(symbol)] for symbol in elements]
    except KeyError as exc:
        raise ValueError(f"unknown element symbol {exc.args[0]!r}") from exc
    return torch.tensor(values, dtype=torch.long)


def _structure_from_jarvis(
    raw: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lattice = torch.as_tensor(raw["lattice_mat"], dtype=torch.float64)
    coordinates = torch.as_tensor(raw["coords"], dtype=torch.float64)
    if bool(raw.get("cartesian", False)):
        coordinates = torch.linalg.solve(lattice.T, coordinates.T).T
    return lattice, coordinates.remainder(1.0), _atomic_numbers(raw["elements"])


def _structure_from_matten(
    raw: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lattice = torch.as_tensor(raw["lattice"]["matrix"], dtype=torch.float64)
    sites = raw["sites"]
    elements = []
    fractional = []
    for site in sites:
        species = site["species"]
        if len(species) != 1 or float(species[0].get("occu", 0.0)) != 1.0:
            raise ValueError("disordered MatTen sites are outside the frozen data contract")
        elements.append(species[0]["element"])
        fractional.append(site["abc"])
    return lattice, torch.as_tensor(fractional, dtype=torch.float64), _atomic_numbers(elements)


def _published_split(raw: Mapping[str, Any]) -> SplitManifest:
    splits = raw["splits"]
    ids = {
        name: tuple(str(row["jarvis_id"]) for row in splits[name])
        for name in ("train", "validation", "test")
    }
    all_ids = ids["train"] + ids["validation"] + ids["test"]
    return SplitManifest(
        seed=32,
        source="published",
        train=ids["train"],
        validation=ids["validation"],
        test=ids["test"],
        sample_to_group={sample_id: sample_id for sample_id in all_ids},
    )


def _make_sample(
    sample_id: str,
    unit: TrainingUnit,
    structure: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    target_unit: str,
    source: Mapping[str, Any],
) -> TensorSample:
    lattice, fractional, numbers = structure
    coefficients = cartesian_to_irreps(target, unit.target)
    return TensorSample(
        sample_id=sample_id,
        unit=unit,
        lattice=lattice,
        fractional_positions=fractional,
        atomic_numbers=numbers,
        target_cartesian=target,
        target_coefficients=coefficients,
        target_unit=target_unit,
        source=source,
    )


def _load_jarvis(
    unit: TrainingUnit,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sample_ids: set[str] | None,
) -> IndependentTensorDataset:
    path = _resource_path(manifest_path, str(manifest["local_file"]))
    _verify_resource(path, size_bytes=manifest["bytes"], sha256=manifest["sha256"])
    # Pickle is deserialized only after matching the frozen trusted resource hash.
    with path.open("rb") as stream:
        records = pickle.load(stream)  # noqa: S301
    split = _published_split(manifest)
    allowed = set(split.train + split.validation + split.test)
    requested = allowed if sample_ids is None else sample_ids
    if not requested <= allowed:
        raise ValueError("requested JARVIS IDs are outside the published split")
    samples = []
    seen = set()
    split_rows = {
        str(row["jarvis_id"]): row
        for name in ("train", "validation", "test")
        for row in manifest["splits"][name]
    }
    for raw in records:
        sample_id = str(raw["JARVIS_ID"])
        if sample_id not in requested:
            continue
        if sample_id in seen:
            raise ValueError(f"duplicate JARVIS ID {sample_id}")
        seen.add(sample_id)
        if unit.target == "dielectric":
            target = torch.as_tensor(raw["dielectric"], dtype=torch.float64)
            if target.shape != (3, 3) or float(target.abs().max()) >= 100.0:
                raise ValueError(
                    f"manifest-selected dielectric sample {sample_id} fails its filter"
                )
            target_unit = "dimensionless"
        else:
            voigt_gpa = torch.as_tensor(raw["elastic_total_kbar"], dtype=torch.float64) / 10.0
            if voigt_gpa.shape != (6, 6) or float(voigt_gpa.abs().max()) >= 1500.0:
                raise ValueError(f"manifest-selected elastic sample {sample_id} fails its filter")
            if int(manifest.get("elastic_protocol_version", 0)) == 1:
                row = split_rows[sample_id]
                if "elastic_support_mask_bits" not in row:
                    raise ValueError("GMTNet elastic protocol row lacks its support mask")
                bits = int(row["elastic_support_mask_bits"])
                support = torch.tensor(
                    [bool(bits & (1 << index)) for index in range(36)]
                ).reshape(6, 6)
                voigt_gpa = voigt_gpa * support
            target = voigt_stiffness_to_cartesian(voigt_gpa)
            target_unit = "GPa"
        samples.append(
            _make_sample(
                sample_id,
                unit,
                _structure_from_jarvis(raw["atoms"]),
                target,
                target_unit,
                {"resource": path.name, "source_index_id": sample_id},
            )
        )
    missing = requested - seen
    if missing:
        raise ValueError(
            f"published split IDs are missing from the JARVIS resource: {sorted(missing)}"
        )
    return IndependentTensorDataset(unit, samples, split)


def _load_matten(
    unit: TrainingUnit,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sample_ids: set[str] | None,
) -> IndependentTensorDataset:
    path = _resource_path(manifest_path, str(manifest["local_file"]))
    _verify_resource(path, size_bytes=manifest["bytes"], sha256=manifest["sha256"])
    raw = json.loads(path.read_text(encoding="utf-8"))
    split_indices = manifest["split_indices"]
    ids = {
        "train": tuple(f"matten-{index}" for index in split_indices["train"]),
        "validation": tuple(f"matten-{index}" for index in split_indices["val"]),
        "test": tuple(f"matten-{index}" for index in split_indices["test"]),
    }
    all_ids = ids["train"] + ids["validation"] + ids["test"]
    split = SplitManifest(
        seed=0,
        source="published",
        train=ids["train"],
        validation=ids["validation"],
        test=ids["test"],
        sample_to_group={sample_id: sample_id for sample_id in all_ids},
    )
    requested = set(all_ids) if sample_ids is None else sample_ids
    if not requested <= set(all_ids):
        raise ValueError("requested MatTen IDs are outside the published split")
    samples = []
    for sample_id in all_ids:
        if sample_id not in requested:
            continue
        index = sample_id.removeprefix("matten-")
        target = torch.as_tensor(raw["elastic_tensor"][index], dtype=torch.float64)
        samples.append(
            _make_sample(
                sample_id,
                unit,
                _structure_from_matten(raw["structure"][index]),
                target,
                "GPa",
                {
                    "resource": path.name,
                    "row_index": int(index),
                    "formula": raw["formula_pretty"][index],
                },
            )
        )
    return IndependentTensorDataset(unit, samples, split)


def _split_from_json(raw: Mapping[str, Any], sample_ids: tuple[str, ...]) -> SplitManifest:
    if "splits" not in raw:
        return make_seeded_split(sample_ids, sample_ids)
    splits = raw["splits"]
    train = tuple(str(value) for value in splits["train"])
    validation = tuple(str(value) for value in splits["validation"])
    test = tuple(str(value) for value in splits["test"])
    all_ids = train + validation + test
    if set(all_ids) != set(sample_ids):
        raise ValueError("BEC saved split does not cover exactly the processed records")
    return SplitManifest(
        seed=int(splits.get("seed", 20260911)),
        source=str(splits.get("source", "seeded_8_1_1")),
        train=train,
        validation=validation,
        test=test,
        sample_to_group={sample_id: sample_id for sample_id in all_ids},
    )


def _load_bec(
    unit: TrainingUnit,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sample_ids: set[str] | None,
) -> IndependentTensorDataset:
    output = manifest["processed_output"]
    if "size_bytes" not in output or "sha256" not in output:
        raise FileNotFoundError("JARVIS-DFPT processed output is not finalized in the manifest")
    path = _resource_path(manifest_path, str(output["path"]).removeprefix("data/"))
    _verify_resource(path, size_bytes=output["size_bytes"], sha256=output["sha256"])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    record_ids = tuple(str(record["sample_id"]) for record in records)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("JARVIS-DFPT processed sample IDs must be unique")
    split = _split_from_json(output, record_ids)
    requested = set(record_ids) if sample_ids is None else sample_ids
    if not requested <= set(record_ids):
        raise ValueError("requested BEC IDs are absent from the processed resource")
    samples = []
    for raw in records:
        sample_id = str(raw["sample_id"])
        if sample_id not in requested:
            continue
        target = torch.as_tensor(raw["born_effective_charge_e"], dtype=torch.float64)
        samples.append(
            _make_sample(
                sample_id,
                unit,
                (
                    torch.as_tensor(raw["lattice_angstrom"], dtype=torch.float64),
                    torch.as_tensor(raw["fractional_coordinates"], dtype=torch.float64),
                    _atomic_numbers(raw["elements"]),
                ),
                target,
                "elementary_charge",
                {
                    **dict(raw.get("source", {})),
                    "quality": dict(raw.get("quality", {})),
                    "resource": path.name,
                },
            )
        )
    return IndependentTensorDataset(unit, samples, split)


def _dtnet_split(manifest: Mapping[str, Any]) -> SplitManifest:
    expected_target = (
        "dielectric_symmetric.total = (dielectric_raw.total + transpose) / 2"
    )
    if manifest.get("training_target") != expected_target:
        raise ValueError("DTNet manifest has an unsupported training target convention")
    splits = manifest["splits"]
    ids = {
        name: tuple(str(value) for value in splits[name])
        for name in ("train", "validation", "test")
    }
    all_ids = ids["train"] + ids["validation"] + ids["test"]
    counts = {name: len(values) for name, values in ids.items()}
    if counts != {name: int(value) for name, value in manifest["split_counts"].items()}:
        raise ValueError("DTNet split counts differ from the manifest")
    expected_records = int(manifest["processed_output"]["records"])
    if len(all_ids) != int(manifest["filtered_records"]) or len(all_ids) != expected_records:
        raise ValueError("DTNet split does not cover the declared processed records")
    return SplitManifest(
        seed=int(manifest["split_seed"]),
        source="published",
        train=ids["train"],
        validation=ids["validation"],
        test=ids["test"],
        sample_to_group={sample_id: sample_id for sample_id in all_ids},
    )


def _load_dtnet(
    unit: TrainingUnit,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sample_ids: set[str] | None,
) -> IndependentTensorDataset:
    output = manifest["processed_output"]
    path = _resource_path(manifest_path, str(output["path"]))
    _verify_resource(path, size_bytes=output["size_bytes"], sha256=output["sha256"])
    split = _dtnet_split(manifest)
    allowed = set(split.train + split.validation + split.test)
    requested = allowed if sample_ids is None else sample_ids
    if not requested <= allowed:
        raise ValueError("requested DTNet IDs are outside the published split")
    samples = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if int(raw.get("schema_version", 0)) != 1:
                raise ValueError(f"DTNet row {line_number} has an unsupported schema")
            sample_id = str(raw["sample_id"])
            if sample_id not in allowed:
                raise ValueError(f"DTNet processed row {sample_id!r} is outside the manifest split")
            if sample_id in seen:
                raise ValueError(f"duplicate DTNet sample ID {sample_id}")
            seen.add(sample_id)
            if sample_id not in requested:
                continue
            target = torch.as_tensor(
                raw["dielectric_symmetric"]["total"], dtype=torch.float64
            )
            raw_total = torch.as_tensor(raw["dielectric_raw"]["total"], dtype=torch.float64)
            if target.shape != (3, 3) or not torch.isfinite(target).all():
                raise ValueError(f"DTNet sample {sample_id} has an invalid total tensor")
            if not torch.allclose(target, target.T, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"DTNet sample {sample_id} target is not symmetric")
            expected_target = 0.5 * (raw_total + raw_total.T)
            if raw_total.shape != (3, 3) or not torch.allclose(
                target, expected_target, atol=1.0e-12, rtol=0.0
            ):
                raise ValueError(f"DTNet sample {sample_id} violates target symmetrization")
            samples.append(
                _make_sample(
                    sample_id,
                    unit,
                    (
                        torch.as_tensor(raw["lattice_angstrom"], dtype=torch.float64),
                        torch.as_tensor(raw["fractional_coordinates"], dtype=torch.float64),
                        _atomic_numbers(raw["elements"]),
                    ),
                    target,
                    "dimensionless",
                    {
                        "resource": path.name,
                        "band_gap_ev": raw["band_gap_ev"],
                        "dielectric_raw": raw["dielectric_raw"],
                        "quality": raw["quality"],
                        **dict(raw.get("source", {})),
                    },
                )
            )
    missing = allowed - seen
    if missing:
        raise ValueError(f"DTNet processed resource is missing manifest IDs: {sorted(missing)}")
    return IndependentTensorDataset(unit, samples, split)


def load_training_dataset(
    unit: TrainingUnit,
    *,
    manifest_path: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
) -> IndependentTensorDataset:
    """Load one independently split real tensor dataset through its frozen manifest."""

    path = (
        DEFAULT_DATA_MANIFESTS[(unit.dataset, unit.target)]
        if manifest_path is None
        else manifest_path
    )
    resolved, manifest = _load_manifest(path)
    requested = None if sample_ids is None else set(sample_ids)
    if requested is not None and len(requested) != len(sample_ids):
        raise ValueError("requested sample IDs must be unique")
    if unit.dataset == "jarvis_tensor":
        return _load_jarvis(unit, resolved, manifest, requested)
    if unit.dataset == "dtnet":
        return _load_dtnet(unit, resolved, manifest, requested)
    if unit.dataset == "matten":
        return _load_matten(unit, resolved, manifest, requested)
    return _load_bec(unit, resolved, manifest, requested)


def load_five_structure_smoke(
    unit: TrainingUnit,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[TensorSample, ...]:
    """Load the frozen first 3/1/1 records without changing the full split contract."""

    path = (
        DEFAULT_DATA_MANIFESTS[(unit.dataset, unit.target)]
        if manifest_path is None
        else manifest_path
    )
    resolved, manifest = _load_manifest(path)
    if unit.dataset == "jarvis_tensor":
        split = _published_split(manifest)
        ids = (*split.train[:3], split.validation[0], split.test[0])
    elif unit.dataset == "dtnet":
        split = _dtnet_split(manifest)
        ids = (*split.train[:3], split.validation[0], split.test[0])
    elif unit.dataset == "matten":
        indices = manifest["split_indices"]
        ids = tuple(
            f"matten-{value}"
            for value in (*indices["train"][:3], indices["val"][0], indices["test"][0])
        )
    else:
        output = manifest["processed_output"]
        if "size_bytes" not in output or "sha256" not in output:
            raise FileNotFoundError("JARVIS-DFPT processed output is not finalized in the manifest")
        dataset = load_training_dataset(unit, manifest_path=resolved)
        return dataset.five_structure_smoke()
    dataset = load_training_dataset(unit, manifest_path=resolved, sample_ids=ids)
    return dataset.five_structure_smoke()


def load_structure_candidates(
    unit: TrainingUnit,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[StructureCandidate, ...]:
    """Load equilibrium structures without converting targets for fixture selection."""

    if unit.dataset == "jarvis_dfpt":
        raise ValueError("incomplete JARVIS-DFPT is not a 32-PG fixture source")
    path = (
        DEFAULT_DATA_MANIFESTS[(unit.dataset, unit.target)]
        if manifest_path is None
        else manifest_path
    )
    resolved, manifest = _load_manifest(path)
    resource = _resource_path(resolved, str(manifest["local_file"]))
    _verify_resource(resource, size_bytes=manifest["bytes"], sha256=manifest["sha256"])
    manifest_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
    candidates = []
    if unit.dataset == "jarvis_tensor":
        with resource.open("rb") as stream:
            records = pickle.load(stream)  # noqa: S301
        split = _published_split(manifest)
        allowed = set(split.train + split.validation + split.test)
        for raw in records:
            sample_id = str(raw["JARVIS_ID"])
            if sample_id not in allowed:
                continue
            lattice, fractional, numbers = _structure_from_jarvis(raw["atoms"])
            candidates.append(
                StructureCandidate(
                    sample_id,
                    unit.namespace,
                    manifest_sha,
                    lattice,
                    fractional,
                    numbers,
                )
            )
    elif unit.dataset == "matten":
        raw = json.loads(resource.read_text(encoding="utf-8"))
        indices = (
            *manifest["split_indices"]["train"],
            *manifest["split_indices"]["val"],
            *manifest["split_indices"]["test"],
        )
        for index_value in indices:
            index = str(index_value)
            lattice, fractional, numbers = _structure_from_matten(raw["structure"][index])
            candidates.append(
                StructureCandidate(
                    f"matten-{index}",
                    unit.namespace,
                    manifest_sha,
                    lattice,
                    fractional,
                    numbers,
                )
            )
    else:
        dataset = load_training_dataset(unit, manifest_path=resolved)
        for sample in dataset:
            candidates.append(
                StructureCandidate(
                    sample.sample_id,
                    unit.namespace,
                    manifest_sha,
                    sample.lattice,
                    sample.fractional_positions,
                    sample.atomic_numbers,
                )
            )
    ids = [candidate.sample_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("fixture-source candidate IDs must be unique within a dataset")
    return tuple(candidates)
