"""Independent numerical audit and incomplete-report checks with synthetic fixtures."""
import copy
import json

import numpy as np
import pytest

from good_result.pretrain.report import audit_run, mean_std, table
from src.experiments.pretrain.protocol import nested_ids, sha256, write_json


def fixture_run(tmp_path):
    rows = [{"record_id": f"{split}{i}", "split": split, "tensor": (np.eye(3)*10).tolist()}
            for split,n in (("train",4),("validation",1),("test",2)) for i in range(n)]
    spec={"task":"dielectric","fraction":100,"seed":42,"model":"pretrain"}
    provenance={"fixture":True}
    target=np.eye(3,dtype=np.float32)*10
    predicted=target.copy(); predicted[0,0]+=3
    records=[{"sample_id":"test0","target":target.tolist(),"prediction":predicted.tolist()},
             {"sample_id":"test1","target":target.tolist(),"prediction":target.tolist()}]
    pred_path=tmp_path/"predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(r) for r in records))
    history=[{"epoch":1,"validation_fnorm":2.0},{"epoch":2,"validation_fnorm":1.0}]
    report={"status":"passed","smoke":False,"spec":spec,"provenance":provenance,
            "split_ids":nested_ids(rows,100),"epochs":2,"history":history,"best_epoch":2,
            "prediction_sha256":sha256(pred_path),
            "test_metrics":{"fnorm":1.5,"rmse":np.sqrt(.5),"mae":1/6,
                            "ewt_5":50,"ewt_10":50,"ewt_25":100},
            "timing":{"feature_preparation_seconds":2.,"graph_preparation_seconds":1.,
                      "training_seconds":10.,"total_seconds":13.,"inference":{"samples":[]}}}
    path=tmp_path/"summary.json"
    write_json(path,report)
    return path,spec,provenance,rows,report


def test_independent_metrics_and_risk(tmp_path):
    path,spec,prov,rows,_=fixture_run(tmp_path)
    got=audit_run(path,spec,prov,rows)
    assert got["_distance"].tolist()==[3.,0.]
    assert got["risk"]["over_25pct"]==0


@pytest.mark.parametrize("field",["fnorm","subset","time","checkpoint"])
def test_audit_rejects_corruption(tmp_path,field):
    path,spec,prov,rows,report=fixture_run(tmp_path)
    if field=="fnorm": report["test_metrics"]["fnorm"]=2
    if field=="subset": report["split_ids"]["train"].pop()
    if field=="time": report["timing"]["total_seconds"]=12
    if field=="checkpoint": report["best_epoch"]=1
    write_json(path,report)
    with pytest.raises(ValueError):
        audit_run(path,spec,prov,rows)


def test_pending_table_does_not_invent_metrics():
    markup,rows=table([])
    assert len(rows)==16
    assert all(row["Fnorm"] is None and row["seed_count"]==0 for row in rows)
    assert "Pending" in markup
    assert "DPA" not in markup and "GMTNet" not in markup


def test_std_is_sample_std_and_single_seed_unknown():
    assert mean_std([3])[1] is None
    assert mean_std([1,2,3])==(2.,1.)
