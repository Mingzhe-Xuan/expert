import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from prepare_jarvis_bec import load_dfpt_index, parse_vasprun_xml  # noqa: E402


def test_parse_minimal_vasprun_extracts_atom_aligned_bec():
    xml = b"""<modeling>
      <atominfo><array name="atoms"><set>
        <rc><c>Na</c><c>1</c></rc><rc><c>Cl</c><c>2</c></rc>
      </set></array></atominfo>
      <calculation><array name="born_charges">
        <set><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></set>
        <set><v>-1 0 0</v><v>0 -1 0</v><v>0 0 -1</v></set>
      </array></calculation>
      <structure name="finalpos"><crystal><varray name="basis">
        <v>4 0 0</v><v>0 4 0</v><v>0 0 4</v>
      </varray></crystal><varray name="positions">
        <v>0 0 0</v><v>.5 .5 .5</v>
      </varray></structure>
    </modeling>"""

    record = parse_vasprun_xml(xml, "JVASP-test")

    assert record["sample_id"] == "JVASP-test"
    assert record["elements"] == ["Na", "Cl"]
    assert len(record["born_effective_charge_e"]) == 2
    assert record["quality"]["acoustic_sum_rule_frobenius_e"] == 0.0


def test_load_dfpt_index(tmp_path):
    payload = {
        "DFPT": [
            {
                "name": "JVASP-1.zip",
                "id": 1,
                "download_url": "https://example.invalid/1",
                "size": 123,
                "supplied_md5": hashlib.md5(b"x").hexdigest(),
            }
        ]
    }
    archive_path = tmp_path / "index.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "figshare_data-10-28-2020.json", json.dumps(payload)
        )

    assert load_dfpt_index(archive_path) == payload["DFPT"]
