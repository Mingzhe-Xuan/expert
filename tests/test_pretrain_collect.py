import io
import tarfile

import pytest

from src.experiments.pretrain.collect import extract_delivery


def archive_with(path,name,*,symlink=False):
    with tarfile.open(path,"w:gz") as tar:
        member=tarfile.TarInfo(name)
        payload=b"{}"
        member.size=len(payload)
        if symlink:
            member.type=tarfile.SYMTYPE
            member.linkname="../../elsewhere"
            member.size=0
        tar.addfile(member,None if symlink else io.BytesIO(payload))


def test_accept_only_report_delivery(tmp_path):
    archive=tmp_path/"delivery.tar.gz"
    archive_with(archive,"good_result/pretrain/metrics.json")
    extract_delivery(archive,tmp_path)
    assert (tmp_path/"good_result/pretrain/metrics.json").read_text()=="{}"


@pytest.mark.parametrize("name,symlink",[("../outside.json",False),("good_result/pretrain/script.py",False),
                                        (".git/config",False),("good_result/pretrain/linked.json",True),
                                        ("results/unrelated/result.json",False)])
def test_reject_unexpected_delivery_members(tmp_path,name,symlink):
    archive=tmp_path/"delivery.tar.gz"
    archive_with(archive,name,symlink=symlink)
    with pytest.raises(ValueError):
        extract_delivery(archive,tmp_path)
