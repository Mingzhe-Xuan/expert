from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifests" / "backbones.json"
BACKBONE_FAMILIES = ("mace", "grace", "dpa4", "equiformerv2")


@dataclass(frozen=True, slots=True)
class BackboneResource:
    family: str
    repository: str
    revision: str
    filename: str
    local_path: Path
    required_runtime: str
    pretraining_dataset: str
    feature_tap: str
    feature_layout: str
    lmax: int
    cutoff_angstrom: float
    precision: str
    elements: str
    parity_policy: str
    graph_policy: str
    license: str
    status: str
    size_bytes: int | None = None
    sha256: str | None = None

    @property
    def available(self) -> bool:
        return self.status.startswith("downloaded_")

    def verify(self) -> Path:
        if not self.available:
            raise FileNotFoundError(
                f"{self.family} checkpoint is unavailable: status={self.status}"
            )
        if self.size_bytes is None or self.sha256 is None:
            raise ValueError(f"available {self.family} resource lacks size/checksum")
        if not self.local_path.is_file():
            raise FileNotFoundError(f"missing {self.family} checkpoint: {self.local_path}")
        if self.local_path.stat().st_size != self.size_bytes:
            raise ValueError(f"{self.family} checkpoint size mismatch")
        digest = hashlib.sha256()
        with self.local_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.sha256:
            raise ValueError(f"{self.family} checkpoint SHA-256 mismatch")
        return self.local_path


class BackboneResourceRegistry:
    """Strict, path-safe registry for the four frozen acceptance checkpoints."""

    def __init__(
        self,
        path: str | Path = DEFAULT_MANIFEST,
        *,
        workspace_root: str | Path | None = None,
    ) -> None:
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported backbone manifest schema")
        records = payload.get("selected_backbones")
        if not isinstance(records, dict) or tuple(records) != BACKBONE_FAMILIES:
            raise ValueError("manifest must contain the four frozen backbones in canonical order")
        workspace = (
            manifest_path.parents[2]
            if workspace_root is None
            else Path(workspace_root).resolve()
        )
        resources = {}
        required = (
            "repository", "revision", "file", "local_path", "required_runtime",
            "pretraining_dataset", "feature_tap", "feature_layout", "lmax",
            "cutoff_angstrom", "precision", "elements", "parity_policy",
            "graph_policy", "license", "status",
        )
        for family in BACKBONE_FAMILIES:
            raw = records[family]
            missing = [key for key in required if key not in raw or raw[key] in (None, "")]
            if missing:
                raise ValueError(f"{family} manifest entry lacks {missing}")
            local_path = (workspace / str(raw["local_path"])).resolve()
            if workspace != local_path and workspace not in local_path.parents:
                raise ValueError(f"{family} local_path escapes the workspace")
            parity = str(raw["parity_policy"])
            expected_parity = (
                "inversion_paired_reynolds"
                if family in {"dpa4", "equiformerv2"}
                else None
            )
            if expected_parity is not None and parity != expected_parity:
                raise ValueError(f"{family} requires inversion-paired Reynolds parity")
            resources[family] = BackboneResource(
                family=family,
                repository=str(raw["repository"]),
                revision=str(raw["revision"]),
                filename=str(raw["file"]),
                local_path=local_path,
                required_runtime=str(raw["required_runtime"]),
                pretraining_dataset=str(raw["pretraining_dataset"]),
                feature_tap=str(raw["feature_tap"]),
                feature_layout=str(raw["feature_layout"]),
                lmax=int(raw["lmax"]),
                cutoff_angstrom=float(raw["cutoff_angstrom"]),
                precision=str(raw["precision"]),
                elements=str(raw["elements"]),
                parity_policy=parity,
                graph_policy=str(raw["graph_policy"]),
                license=str(raw["license"]),
                status=str(raw["status"]),
                size_bytes=None if raw.get("size_bytes") is None else int(raw["size_bytes"]),
                sha256=None if raw.get("sha256") is None else str(raw["sha256"]).lower(),
            )
        self.path = manifest_path
        self._resources = resources

    def __getitem__(self, family: str) -> BackboneResource:
        try:
            return self._resources[family]
        except KeyError as exc:
            raise KeyError(f"unknown backbone family {family!r}") from exc

    def __iter__(self):
        return (self._resources[family] for family in BACKBONE_FAMILIES)
