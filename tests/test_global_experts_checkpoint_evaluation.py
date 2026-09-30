from dataclasses import asdict
import json

import pytest
import torch

from src.training.global_experts import checkpoint_evaluation as evaluation
from src.training.global_experts.runner import GlobalExpertsTrainConfig


def source_history():
    return {"history": [{"epoch": n, "validation_mae": float(201-n)}
                        for n in range(1, 201)],
            "training_config": {"epochs": 200}, "best_epoch": 14}


def test_selection_retained_only_exclusive_and_ties(tmp_path):
    source = source_history()
    source["history"][99]["validation_mae"] = -10.
    source["history"][116]["validation_mae"] = -5.
    for epoch in (100, 120, 160, 200):
        (tmp_path / f"epoch-{epoch:04d}.pt").touch()
    selected, path, historical = evaluation.select_retained(source, tmp_path)
    assert selected["epoch"] == 200 and path.name == "epoch-0200.pt"
    assert historical["epoch"] == 117
    source["history"][159]["validation_mae"] = 1.
    assert evaluation.select_retained(source, tmp_path)[0]["epoch"] == 160
    with pytest.raises(ValueError, match="no retained"):
        evaluation.select_retained(source, tmp_path, 200)


def test_invalid_history(tmp_path):
    source = source_history()
    source["history"][0]["validation_mae"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        evaluation.select_retained(source, tmp_path)
    source["history"].pop()
    with pytest.raises(ValueError, match="noncontiguous"):
        evaluation.select_retained(source, tmp_path)


@pytest.mark.parametrize("corruption", [None, "sha", "epoch", "metadata", "step"])
@pytest.mark.parametrize("legacy", [False, True])
def test_evaluate_without_optimizer(tmp_path, monkeypatch, corruption, legacy):
    class Model(torch.nn.Linear):
        def metadata(self):
            return {"test": True, "expert_numbers": (5, 8)}
    model = Model(1, 1)
    config = GlobalExpertsTrainConfig(batch_size=1, minimum_checkpoint_epoch_exclusive=0)
    stored_config = asdict(config)
    if legacy:
        del stored_config["decay_epochs"], stored_config["minimum_checkpoint_epoch_exclusive"]
    splits = {name: [{"sample_id": name, "target": torch.eye(3)}]
              for name in ("train", "validation", "test")}
    source = {**source_history(), "status": "passed", "provenance": {},
              "model_metadata": model.metadata(), "training_config": stored_config,
              "split_counts": {k: 1 for k in splits}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    checkpoint = tmp_path / "epoch-0200.pt"
    torch.save({"schema_version": 1, "model_metadata": (
        {} if corruption == "metadata" else model.metadata()), "provenance": {},
        "model_state": model.state_dict(), "training_config": stored_config,
        "epoch": 100 if corruption == "epoch" else 200,
        "step": 0 if corruption == "step" else 200}, checkpoint)
    before = checkpoint.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer must never be constructed")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(evaluation, "predict", lambda *a: torch.eye(3).unsqueeze(0))
    args = dict(output_dir=tmp_path / "evaluation", provenance={}, config=config,
                device="cpu", source_summary=path,
                source_sha256="bad" if corruption == "sha" else evaluation.file_sha256(path))
    if corruption:
        with pytest.raises(ValueError, match="mismatch"):
            evaluation.evaluate_checkpoint(model, splits, **args)
        assert not (tmp_path / "evaluation").exists()
    else:
        result = evaluation.evaluate_checkpoint(model, splits, **args)
        assert result["best_epoch"] == 200 and result["optimizer_steps"] == 0
        assert result["test_metrics"]["rmse"] == 0
        assert (tmp_path / "evaluation" / "predictions.jsonl").is_file()
        with pytest.raises(FileExistsError):
            evaluation.evaluate_checkpoint(model, splits, **args)
    assert checkpoint.read_bytes() == before
