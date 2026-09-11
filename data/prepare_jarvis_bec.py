"""Download and extract atom-resolved JARVIS-DFPT Born effective charges.

The official JARVIS raw-file index contains one ZIP archive per DFPT
calculation.  This utility verifies every archive against the Figshare MD5,
then extracts the calculation-matched final structure and the complete
``N x 3 x 3`` Born effective charge tensor from ``vasprun.xml``.

Only Python's standard library is required.  The output is JSON Lines so an
interrupted run can be resumed without holding the full dataset in memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INDEX_MEMBER = "figshare_data-10-28-2020.json"


def _vectors(parent: ET.Element, path: str) -> list[list[float]]:
    node = parent.find(path)
    if node is None:
        raise ValueError(f"missing XML node: {path}")
    rows = []
    for child in node.findall("v"):
        if child.text is None:
            raise ValueError(f"empty vector in XML node: {path}")
        rows.append([float(value) for value in child.text.split()])
    return rows


def parse_vasprun_xml(xml_bytes: bytes, sample_id: str) -> dict[str, Any]:
    """Extract the final structure and full BEC tensor from a VASP XML file."""

    root = ET.fromstring(xml_bytes)

    atom_set = root.find("./atominfo/array[@name='atoms']/set")
    if atom_set is None:
        raise ValueError("missing atominfo/atoms")
    elements = []
    for row in atom_set.findall("rc"):
        symbol = row.find("c")
        if symbol is None or symbol.text is None:
            raise ValueError("missing element in atominfo/atoms")
        elements.append(symbol.text.strip())

    final_structure = root.find("./structure[@name='finalpos']")
    if final_structure is None:
        raise ValueError("missing finalpos structure")
    lattice = _vectors(final_structure, "./crystal/varray[@name='basis']")
    frac_coords = _vectors(final_structure, "./varray[@name='positions']")

    born_array = root.find(".//array[@name='born_charges']")
    if born_array is None:
        raise ValueError("missing born_charges")
    born_charges = []
    ion_sets = born_array.findall("set")
    if not ion_sets:
        raise ValueError("born_charges has no ion sets")
    # VASP writes one direct ``set`` child per ion.  Accept an additional
    # wrapper as well because some XML conversion tools insert one.
    if len(ion_sets) == 1 and ion_sets[0].find("set") is not None:
        ion_sets = ion_sets[0].findall("set")
    for ion_set in ion_sets:
        tensor = []
        for vector in ion_set.findall("v"):
            if vector.text is None:
                raise ValueError("empty row in born_charges")
            tensor.append([float(value) for value in vector.text.split()])
        born_charges.append(tensor)

    natoms = len(elements)
    if len(lattice) != 3 or any(len(row) != 3 for row in lattice):
        raise ValueError(f"invalid lattice shape for {sample_id}")
    if len(frac_coords) != natoms or any(len(row) != 3 for row in frac_coords):
        raise ValueError(f"coordinate/atom mismatch for {sample_id}")
    if len(born_charges) != natoms or any(
        len(tensor) != 3 or any(len(row) != 3 for row in tensor)
        for tensor in born_charges
    ):
        raise ValueError(f"BEC/atom mismatch for {sample_id}")
    flat_values = (
        value
        for tensor in born_charges
        for row in tensor
        for value in row
    )
    if not all(math.isfinite(value) for value in flat_values):
        raise ValueError(f"non-finite BEC for {sample_id}")

    asr_matrix = [
        [sum(tensor[i][j] for tensor in born_charges) for j in range(3)]
        for i in range(3)
    ]
    asr_frobenius = math.sqrt(
        sum(value * value for row in asr_matrix for value in row)
    )
    return {
        "sample_id": sample_id,
        "elements": elements,
        "lattice_angstrom": lattice,
        "fractional_coordinates": frac_coords,
        "born_effective_charge_e": born_charges,
        "quality": {
            "natoms": natoms,
            "acoustic_sum_rule_matrix_e": asr_matrix,
            "acoustic_sum_rule_frobenius_e": asr_frobenius,
        },
    }


def load_dfpt_index(index_zip: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(index_zip) as archive:
        payload = json.loads(archive.read(DEFAULT_INDEX_MEMBER))
    records = payload.get("DFPT")
    if not isinstance(records, list):
        raise ValueError("raw-file index has no DFPT list")
    return records


def _download(url: str, destination: Path, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "expert-jarvis-bec-preparer/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            os.replace(temporary, destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify upstream Figshare MD5
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_one(
    source: dict[str, Any], archive_dir: Path, keep_archives: bool
) -> dict[str, Any]:
    filename = str(source["name"])
    sample_id = filename.removesuffix(".zip")
    archive_path = archive_dir / filename
    expected_md5 = str(source["supplied_md5"]).lower()
    if not archive_path.exists() or _file_md5(archive_path) != expected_md5:
        archive_path.unlink(missing_ok=True)
        _download(str(source["download_url"]), archive_path)
    actual_md5 = _file_md5(archive_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {sample_id}: {actual_md5} != {expected_md5}"
        )
    with zipfile.ZipFile(archive_path) as archive:
        xml_bytes = archive.read("vasprun.xml")
    record = parse_vasprun_xml(xml_bytes, sample_id)
    record["source"] = {
        "archive_name": filename,
        "figshare_file_id": int(source["id"]),
        "download_url": str(source["download_url"]),
        "archive_size_bytes": int(source["size"]),
        "archive_md5": actual_md5,
        "vasprun_sha256": hashlib.sha256(xml_bytes).hexdigest(),
    }
    if not keep_archives:
        archive_path.unlink()
    return record


def _completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                completed.add(str(json.loads(line)["sample_id"]))
            except Exception as exc:
                raise ValueError(
                    f"invalid existing JSONL at line {line_number}"
                ) from exc
    return completed


def prepare_dataset(
    sources: Iterable[dict[str, Any]],
    output_path: Path,
    archive_dir: Path,
    error_path: Path,
    workers: int,
    keep_archives: bool,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output_path)
    pending = [
        item
        for item in sources
        if str(item["name"]).removesuffix(".zip") not in completed
    ]

    successes = 0
    failures = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as output:
        # Errors describe the current resumable attempt. Resolved transient failures from
        # an earlier attempt must not permanently invalidate a subsequently complete dataset.
        with error_path.open("w", encoding="utf-8", buffering=1) as errors:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                futures = {
                    executor.submit(
                        _prepare_one, item, archive_dir, keep_archives
                    ): item
                    for item in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    item = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        failures += 1
                        errors.write(
                            json.dumps(
                                {
                                    "archive_name": item.get("name"),
                                    "download_url": item.get("download_url"),
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    else:
                        successes += 1
                        output.write(
                            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                    done = successes + failures
                    if done % 100 == 0 or done == len(pending):
                        print(
                            f"processed={done}/{len(pending)} "
                            f"success={successes} failure={failures}",
                            flush=True,
                        )
    return successes, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    sources = load_dfpt_index(args.index_zip)
    if args.limit is not None:
        sources = sources[: args.limit]
    successes, failures = prepare_dataset(
        sources=sources,
        output_path=args.output,
        archive_dir=args.archive_dir,
        error_path=args.errors,
        workers=args.workers,
        keep_archives=args.keep_archives,
    )
    print(f"new_successes={successes} failures={failures}")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
