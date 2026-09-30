from dataclasses import asdict

import pytest
import torch

from src.training.global_experts import runner


def test_constant_tail_matches_every_original_update():
    config = runner.GlobalExpertsTrainConfig(epochs=300, decay_epochs=200)
    steps_per_epoch = 79
    for step in range(200 * steps_per_epoch + 1):
        old = 1e-5 + (1e-3 - 1e-5) * (1 - step / (200 * steps_per_epoch))
        assert runner.learning_rate_after_step(config, step, steps_per_epoch) == old
    for step in range(200 * steps_per_epoch, 300 * steps_per_epoch + 1):
        assert runner.learning_rate_after_step(config, step, steps_per_epoch) == 1e-5
    default = runner.GlobalExpertsTrainConfig()
    assert default.minimum_checkpoint_epoch_exclusive == 100
    for step in (0, 79, 7900, 15800):
        assert runner.learning_rate_after_step(default, step, 79) == runner.learning_rate_after_step(config, step, 79)


@pytest.mark.parametrize("kwargs", [dict(epochs=100), dict(decay_epochs=-1),
    dict(decay_epochs=201), dict(minimum_checkpoint_epoch_exclusive=-1)])
def test_invalid_training_boundaries(kwargs):
    with pytest.raises(ValueError):
        runner.GlobalExpertsTrainConfig(**kwargs)


def test_post100_selects_and_reloads_only_eligible_checkpoint(monkeypatch, tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, 3))
        def metadata(self):
            return {"tiny": True}
        def forward(self, *args):
            return self.weight.unsqueeze(0)
    model = Tiny()
    splits = {name: [{"sample_id": name, "target": torch.eye(3)}]
              for name in ("train", "validation", "test")}
    def collate(rows, device):
        return None, None, None, (), torch.eye(3).unsqueeze(0), torch.ones(1,3,3,dtype=torch.bool)
    monkeypatch.setattr(runner, "collate", collate)
    monkeypatch.setattr(runner, "predict", lambda *args: model().detach())
    scores = iter([0.] * 100 + [2., 3.])
    monkeypatch.setattr(runner, "_validation_history_metrics",
                        lambda *args: {"validation_mae": next(scores)})
    config = runner.GlobalExpertsTrainConfig(epochs=102, decay_epochs=100, checkpoint_interval=0)
    report = runner.train_global_experts(model, splits, output_dir=tmp_path / "run",
        provenance={}, config=config)
    assert report["best_epoch"] == 101
    saved = torch.load(tmp_path / "run/best.pt", weights_only=True)
    assert saved["epoch"] == 101 and saved["training_config"] == asdict(config)
    torch.testing.assert_close(model.weight, saved["model_state"]["weight"])
    assert report["history"][99]["learning_rate"] == 1e-5
    assert report["history"][101]["learning_rate"] == 1e-5


def test_cli_options_and_old_checkpoint_normalization():
    from src.cli.global_experts_train import parser
    args = parser().parse_args(["--official-root", "source", "--graph-cache", "g",
        "--routing-cache", "r", "--output-dir", "out", "--epochs", "300", "--decay-epochs", "200"])
    assert args.decay_epochs == 200 and args.minimum_checkpoint_epoch_exclusive == 100
    saved = asdict(runner.GlobalExpertsTrainConfig())
    del saved["decay_epochs"], saved["minimum_checkpoint_epoch_exclusive"]
    normalized = runner.normalized_training_config(saved)
    assert normalized["decay_epochs"] == normalized["minimum_checkpoint_epoch_exclusive"] == 0
