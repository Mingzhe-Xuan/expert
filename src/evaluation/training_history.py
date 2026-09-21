from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


REQUIRED_HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_mae",
    "validation_fnorm",
    "learning_rate",
)
CURRENT_GROUP_ROUTING_ALIASES = {
    "current_group_only",
    "current_point_group_only",
}
GMTNET_HISTORY_FIELDS = ("epoch", "training_loss", "validation_mae", "learning_rate")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_current_group_history(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_epochs: int = 200,
) -> dict[str, object]:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256.lower():
        raise ValueError("training summary SHA-256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError("training curve requires a passed summary")
    if payload.get("routing") not in CURRENT_GROUP_ROUTING_ALIASES:
        raise ValueError("training curve source is not current-group-only")
    if "CGCNN" not in str(payload.get("model", "")):
        raise ValueError("training curve source is not the CGCNN-style branch")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError(f"training history must contain exactly {expected_epochs} epochs")
    epochs = []
    for row in history:
        if not isinstance(row, Mapping) or any(field not in row for field in REQUIRED_HISTORY_FIELDS):
            raise ValueError("training history row lacks required fields")
        values = [float(row[field]) for field in REQUIRED_HISTORY_FIELDS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("training history contains non-finite values")
        epochs.append(int(row["epoch"]))
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError("training history epochs must be contiguous and one-indexed")
    best_epoch = int(payload.get("best_epoch", 0))
    minimum_epoch = min(history, key=lambda row: float(row["validation_mae"]))["epoch"]
    if best_epoch != int(minimum_epoch):
        raise ValueError("best epoch disagrees with minimum validation MAE")
    return payload


def load_parent_dag_history(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_epochs: int = 200,
) -> dict[str, object]:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256.lower():
        raise ValueError("parent-DAG summary SHA-256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError("training curve requires a passed parent-DAG summary")
    if payload.get("routing") != "point_group_parent_dag_all_ancestors":
        raise ValueError("training curve source is not all-ancestor point-group routing")
    if "PG-parent-DAG all-ancestors" not in str(payload.get("model", "")):
        raise ValueError("training curve source is not the parent-DAG CGCNN branch")
    dag = payload.get("point_group_dag")
    if not isinstance(dag, Mapping) or dag.get("activation") != "current_plus_all_transitive_parents":
        raise ValueError("parent-DAG summary lacks the all-ancestor activation contract")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError(f"parent-DAG history must contain exactly {expected_epochs} epochs")
    epochs = []
    for row in history:
        if not isinstance(row, Mapping) or any(field not in row for field in REQUIRED_HISTORY_FIELDS):
            raise ValueError("parent-DAG history row lacks required fields")
        values = [float(row[field]) for field in REQUIRED_HISTORY_FIELDS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("parent-DAG history contains non-finite values")
        epochs.append(int(row["epoch"]))
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError("parent-DAG history epochs must be contiguous and one-indexed")
    best_epoch = int(payload.get("best_epoch", 0))
    minimum_epoch = min(history, key=lambda row: float(row["validation_mae"]))["epoch"]
    if best_epoch != int(minimum_epoch):
        raise ValueError("parent-DAG best epoch disagrees with minimum validation MAE")
    return payload


def load_gmtnet_history(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_epochs: int = 200,
) -> dict[str, object]:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256.lower():
        raise ValueError("GMTNet summary SHA-256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("model") != "GMTNet":
        raise ValueError("training curve requires a passed GMTNet summary")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError(f"GMTNet history must contain exactly {expected_epochs} epochs")
    epochs = []
    for row in history:
        if not isinstance(row, Mapping) or any(field not in row for field in GMTNET_HISTORY_FIELDS):
            raise ValueError("GMTNet history row lacks required fields")
        values = [float(row[field]) for field in GMTNET_HISTORY_FIELDS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("GMTNet history contains non-finite values")
        epochs.append(int(row["epoch"]))
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError("GMTNet epochs must be contiguous and one-indexed")
    best_epoch = int(payload.get("best_epoch", 0))
    minimum_epoch = min(history, key=lambda row: float(row["validation_mae"]))["epoch"]
    if best_epoch != int(minimum_epoch):
        raise ValueError("GMTNet best epoch disagrees with minimum validation MAE")
    return payload


def render_current_group_history(
    summary: Mapping[str, object],
    *,
    gmtnet_summary: Mapping[str, object] | None = None,
    svg_path: str | Path,
    png_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    history: Sequence[Mapping[str, object]] = summary["history"]  # type: ignore[assignment]
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]
    validation_mae = [float(row["validation_mae"]) for row in history]
    validation_fnorm = [float(row["validation_fnorm"]) for row in history]
    learning_rate = [float(row["learning_rate"]) for row in history]
    best_epoch = int(summary["best_epoch"])
    best_mae = float(summary["best_validation_mae"])
    if gmtnet_summary is not None:
        gmtnet_history: Sequence[Mapping[str, object]] = gmtnet_summary["history"]  # type: ignore[assignment]
        if [int(row["epoch"]) for row in gmtnet_history] != epochs:
            raise ValueError("CGCNN and GMTNet histories must cover identical epochs")
        gmtnet_train_loss = [float(row["training_loss"]) for row in gmtnet_history]
        gmtnet_validation_mae = [float(row["validation_mae"]) for row in gmtnet_history]
        gmtnet_learning_rate = [float(row["learning_rate"]) for row in gmtnet_history]
        gmtnet_best_epoch = int(gmtnet_summary["best_epoch"])
        gmtnet_best_mae = float(gmtnet_summary["best_validation_mae"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "expert-current-group-training-v1",
        }
    )
    colors = {
        "train": "#2E6F9E",
        "validation": "#E07A3F",
        "mae": "#2A9D8F",
        "fnorm": "#8C5DAA",
        "lr": "#53606B",
        "best": "#2A9D8F",
        "gmt": "#D1495B",
    }
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1.25, 0.65), "hspace": 0.16},
    )
    figure.suptitle(
        (
            "current-pg vs GMTNet — training history"
            if gmtnet_summary is not None
            else "CGCNN-style feature + current-group only — training history"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.951,
        (
            "Reduced dielectric-total · current-pg job 458 · GMTNet job 443 · 200 epochs"
            if gmtnet_summary is not None
            else "Reduced dielectric-total · B+A+PGE+R/full_pg · job 458 · 200 epochs"
        ),
        ha="center",
        color="#53606B",
    )

    axes[0].plot(
        epochs,
        train_loss,
        color=colors["train"],
        linewidth=1.8,
        label="current-pg train Huber",
    )
    axes[0].plot(
        epochs,
        validation_loss,
        color=colors["validation"],
        linewidth=1.8,
        label="current-pg validation Huber",
    )
    if gmtnet_summary is not None:
        axes[0].plot(
            epochs,
            gmtnet_train_loss,
            color=colors["gmt"],
            linewidth=1.8,
            linestyle=(0, (5, 2)),
            label="GMTNet train Huber",
        )
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title("Optimization objective", loc="left")
    axes[0].legend(frameon=False, ncol=3 if gmtnet_summary is not None else 2, loc="upper right")

    axes[1].plot(
        epochs,
        validation_mae,
        color=colors["mae"],
        linewidth=1.8,
        label="current-pg validation MAE",
    )
    if gmtnet_summary is not None:
        axes[1].plot(
            epochs,
            gmtnet_validation_mae,
            color=colors["gmt"],
            linewidth=1.8,
            linestyle=(0, (5, 2)),
            label="GMTNet validation MAE",
        )
    axes[1].set_ylabel("Component MAE")
    axes[1].set_title("Validation metrics", loc="left")
    metric_right = axes[1].twinx()
    metric_right.spines["right"].set_visible(True)
    metric_right.plot(
        epochs,
        validation_fnorm,
        color=colors["fnorm"],
        linewidth=1.6,
        alpha=0.9,
        label="current-pg validation Fnorm",
    )
    metric_right.set_ylabel("Fnorm", color=colors["fnorm"])
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = metric_right.get_legend_handles_labels()
    axes[1].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    axes[1].scatter(
        [best_epoch],
        [best_mae],
        s=48,
        color=colors["best"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    axes[1].annotate(
        f"current-pg best epoch {best_epoch}\nvalidation MAE {best_mae:.4f}",
        xy=(best_epoch, best_mae),
        xytext=(0.98, 0.84),
        textcoords="axes fraction",
        ha="right",
        va="top",
        color=colors["best"],
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9},
        arrowprops={"arrowstyle": "-", "color": colors["best"], "lw": 0.9},
    )
    if gmtnet_summary is not None:
        axes[1].scatter(
            [gmtnet_best_epoch],
            [gmtnet_best_mae],
            s=48,
            color=colors["gmt"],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        axes[1].annotate(
            f"GMTNet best epoch {gmtnet_best_epoch}\nvalidation MAE {gmtnet_best_mae:.4f}",
            xy=(gmtnet_best_epoch, gmtnet_best_mae),
            xytext=(0.55, 0.68),
            textcoords="axes fraction",
            ha="center",
            va="top",
            color=colors["gmt"],
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": colors["gmt"], "lw": 0.9},
        )

    if gmtnet_summary is not None and all(
        math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
        for left, right in zip(learning_rate, gmtnet_learning_rate)
    ):
        lr_label = "Shared LR schedule (both models)"
    else:
        lr_label = "current-pg learning rate"
    axes[2].plot(
        epochs, learning_rate, color=colors["lr"], linewidth=1.8, label=lr_label
    )
    if gmtnet_summary is not None and lr_label != "Shared LR schedule (both models)":
        axes[2].plot(
            epochs,
            gmtnet_learning_rate,
            color=colors["gmt"],
            linewidth=1.6,
            linestyle=(0, (5, 2)),
            label="GMTNet learning rate",
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Per-step linear decay (epoch-end value)", loc="left")
    axes[2].legend(frameon=False, loc="upper right")
    axes[2].annotate(
        f"final {learning_rate[-1]:.0e}",
        xy=(epochs[-1], learning_rate[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        color=colors["lr"],
        fontsize=9,
    )

    for axis in axes:
        axis.axvline(best_epoch, color=colors["best"], linestyle="--", linewidth=1.0, alpha=0.65)
        if gmtnet_summary is not None:
            axis.axvline(
                gmtnet_best_epoch,
                color=colors["gmt"],
                linestyle=(0, (3, 2)),
                linewidth=1.0,
                alpha=0.55,
            )
        axis.grid(axis="y", color="#D7DDE2", linewidth=0.7, alpha=0.75)
        axis.set_xlim(1, epochs[-1])
    axes[2].set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
    figure.subplots_adjust(left=0.10, right=0.90, bottom=0.08, top=0.91)

    svg = Path(svg_path)
    png = Path(png_path)
    if svg.suffix.lower() != ".svg" or png.suffix.lower() != ".png":
        raise ValueError("training curve outputs must be SVG and PNG")
    svg.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        svg,
        format="svg",
        metadata={
            "Date": None,
            "Title": (
                "current-pg vs GMTNet training history"
                if gmtnet_summary is not None
                else "CGCNN-style feature + current-group only training history"
            ),
        },
    )
    # Matplotlib emits spaces at the ends of multiline SVG path records. Normalize them so the
    # tracked documentation asset passes the repository's whitespace gate deterministically.
    svg_text = svg.read_text(encoding="utf-8")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(
        png,
        format="png",
        dpi=180,
        metadata={
            "Title": (
                "current-pg vs GMTNet training history"
                if gmtnet_summary is not None
                else "CGCNN-style feature + current-group only training history"
            )
        },
    )
    plt.close(figure)


def render_routing_comparison_history(
    current_summary: Mapping[str, object],
    parent_summary: Mapping[str, object],
    *,
    svg_path: str | Path,
    png_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    current: Sequence[Mapping[str, object]] = current_summary["history"]  # type: ignore[assignment]
    parent: Sequence[Mapping[str, object]] = parent_summary["history"]  # type: ignore[assignment]
    epochs = [int(row["epoch"]) for row in current]
    if [int(row["epoch"]) for row in parent] != epochs:
        raise ValueError("current-pg and parent-DAG histories must cover identical epochs")

    def series(history: Sequence[Mapping[str, object]], field: str) -> list[float]:
        return [float(row[field]) for row in history]

    current_best = int(current_summary["best_epoch"])
    parent_best = int(parent_summary["best_epoch"])
    current_best_mae = float(current_summary["best_validation_mae"])
    parent_best_mae = float(parent_summary["best_validation_mae"])
    current_lr = series(current, "learning_rate")
    parent_lr = series(parent, "learning_rate")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "expert-parent-dag-training-v1",
        }
    )
    colors = {"current": "#2E6F9E", "parent": "#D1495B", "fnorm": "#7A5195", "lr": "#53606B"}
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1.25, 0.65), "hspace": 0.16},
    )
    figure.suptitle("current-pg vs all-ancestor PG-DAG — training history", fontsize=16, y=0.985)
    figure.text(
        0.5,
        0.951,
        "Reduced dielectric-total · matched CGCNN jobs 458 and 472 · 200 epochs",
        ha="center",
        color="#53606B",
    )

    for label, history, color in (
        ("current-pg", current, colors["current"]),
        ("parent-DAG", parent, colors["parent"]),
    ):
        axes[0].plot(epochs, series(history, "train_loss"), color=color, lw=1.8, label=f"{label} train")
        axes[0].plot(
            epochs,
            series(history, "validation_loss"),
            color=color,
            lw=1.6,
            linestyle=(0, (5, 2)),
            label=f"{label} validation",
        )
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title("Optimization objective", loc="left")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].plot(
        epochs,
        series(current, "validation_mae"),
        color=colors["current"],
        lw=1.8,
        label="current-pg validation MAE",
    )
    axes[1].plot(
        epochs,
        series(parent, "validation_mae"),
        color=colors["parent"],
        lw=1.8,
        label="parent-DAG validation MAE",
    )
    axes[1].set_ylabel("Component MAE")
    axes[1].set_title("Validation metrics", loc="left")
    metric_right = axes[1].twinx()
    metric_right.spines["right"].set_visible(True)
    metric_right.plot(
        epochs,
        series(current, "validation_fnorm"),
        color=colors["current"],
        lw=1.4,
        alpha=0.65,
        linestyle=":",
        label="current-pg validation Fnorm",
    )
    metric_right.plot(
        epochs,
        series(parent, "validation_fnorm"),
        color=colors["parent"],
        lw=1.4,
        alpha=0.65,
        linestyle=":",
        label="parent-DAG validation Fnorm",
    )
    metric_right.set_ylabel("Fnorm", color=colors["fnorm"])
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = metric_right.get_legend_handles_labels()
    axes[1].legend(handles_left + handles_right, labels_left + labels_right, frameon=False, ncol=2)

    for epoch, mae, color, label, location in (
        (current_best, current_best_mae, colors["current"], "current-pg", (0.97, 0.86)),
        (parent_best, parent_best_mae, colors["parent"], "parent-DAG", (0.57, 0.68)),
    ):
        axes[1].scatter([epoch], [mae], s=48, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        axes[1].annotate(
            f"{label} best epoch {epoch}\nvalidation MAE {mae:.4f}",
            xy=(epoch, mae),
            xytext=location,
            textcoords="axes fraction",
            ha="right",
            va="top",
            color=color,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
        )

    shared_lr = all(
        math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
        for left, right in zip(current_lr, parent_lr)
    )
    axes[2].plot(
        epochs,
        current_lr,
        color=colors["lr"],
        lw=1.8,
        label="Shared LR schedule" if shared_lr else "current-pg learning rate",
    )
    if not shared_lr:
        axes[2].plot(epochs, parent_lr, color=colors["parent"], lw=1.6, label="parent-DAG learning rate")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Per-step linear decay (epoch-end value)", loc="left")
    axes[2].legend(frameon=False, loc="upper right")
    axes[2].annotate(
        f"final {current_lr[-1]:.0e}",
        xy=(epochs[-1], current_lr[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        color=colors["lr"],
        fontsize=9,
    )

    for axis in axes:
        axis.axvline(current_best, color=colors["current"], linestyle="--", lw=1.0, alpha=0.55)
        axis.axvline(parent_best, color=colors["parent"], linestyle=(0, (3, 2)), lw=1.0, alpha=0.55)
        axis.grid(axis="y", color="#D7DDE2", linewidth=0.7, alpha=0.75)
        axis.set_xlim(1, epochs[-1])
    axes[2].set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
    figure.subplots_adjust(left=0.10, right=0.90, bottom=0.08, top=0.91)

    svg = Path(svg_path)
    png = Path(png_path)
    if svg.suffix.lower() != ".svg" or png.suffix.lower() != ".png":
        raise ValueError("training curve outputs must be SVG and PNG")
    svg.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    title = "current-pg vs all-ancestor PG-DAG training history"
    figure.savefig(svg, format="svg", metadata={"Date": None, "Title": title})
    svg_text = svg.read_text(encoding="utf-8")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(png, format="png", dpi=180, metadata={"Title": title})
    plt.close(figure)
