"""Render audited paired experiments without substituting missing measurements."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.pretrain.protocol import (  # noqa: E402
    FRACTIONS, SEEDS, TASKS, crystal_system, grid, load_records, nested_ids, run_name, sha256, write_json,
)

LABELS = ("pretrain", "O(3)")
COLORS = {"pretrain": "#009E73", "O(3)": "#0072B2"}


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.dpi": 180})
    return plt


def save_figure(fig, output, name):
    fig.savefig(output / (name+".png"), bbox_inches="tight")
    fig.savefig(output / (name+".svg"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    text = (output / (name+".svg")).read_text(encoding="utf-8")
    if "DPA" in text or "GMTNet" in text:
        raise ValueError("forbidden presentation label")


def dataset_figures(rows, task, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})
    counts = Counter(r["point_group"] for r in rows)
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    axes[0].bar(range(len(ordered)), [counts[k] for k in ordered], color="#4477AA")
    axes[0].set_xticks(range(len(ordered)), ordered, rotation=60, ha="right")
    axes[0].set(xlabel="Point group", ylabel="Structures", title=f"{TASKS[task]} | N = {len(rows):,}")
    systems = ("Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic")
    system_counts = Counter(crystal_system(r["space_group"]) for r in rows)
    bars = axes[1].barh(systems, [system_counts[k] for k in systems], color="#66AABB")
    axes[1].bar_label(bars, padding=3)
    axes[1].set(xlabel="Structures", title="Crystal systems")
    fig.suptitle("JARVIS-DFT | curated source | all train / validation / test structures")
    fig.tight_layout()
    save_figure(fig, output, task+"_dataset_distribution")
    write_json(output / (task+"_dataset_distribution.json"), {
        "total": len(rows), "point_group": counts, "crystal_system": system_counts,
        "split": Counter(r["split"] for r in rows)})


def audit_run(path, spec, provenance, rows):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "passed" or report["smoke"] or report["spec"] != spec:
        raise ValueError(f"invalid completed run: {path}")
    if report["provenance"] != provenance or report["split_ids"] != nested_ids(rows, spec["fraction"]):
        raise ValueError(f"dataset/subset mismatch: {path}")
    history = report["history"]
    if [r["epoch"] for r in history] != list(range(1, report["epochs"]+1)):
        raise ValueError("missing epoch history")
    eligible = [r for r in history if r["epoch"] > report["epochs"]//2]
    best = min(eligible, key=lambda r: r["validation_fnorm"])
    if best["epoch"] != report["best_epoch"]:
        raise ValueError("checkpoint selection does not match validation history")
    pred_path = path.with_name("predictions.jsonl")
    if sha256(pred_path) != report["prediction_sha256"]:
        raise ValueError("prediction checksum mismatch")
    predictions = [json.loads(line) for line in pred_path.read_text().splitlines()]
    if [p["sample_id"] for p in predictions] != report["split_ids"]["test"]:
        raise ValueError("test ID mismatch")
    pred = np.asarray([p["prediction"] for p in predictions], dtype=float)
    target = np.asarray([p["target"] for p in predictions], dtype=float)
    expected_rows = {r["record_id"]: r for r in rows}
    expected = np.asarray([expected_rows[p["sample_id"]]["tensor"] for p in predictions])
    if spec["task"] == "elastic":
        pairs = ((0,0),(1,1),(2,2),(0,1),(1,2),(0,2))
        expected = np.stack([np.stack([expected[:,i,j,k,l] for k,l in pairs], -1) for i,j in pairs], -2)
    if not np.allclose(target, expected.astype(np.float32), atol=1e-6, rtol=1e-6):
        raise ValueError("test targets differ from curated source")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite predictions")
    error = (pred-target).reshape(len(pred), -1)
    distance = np.linalg.norm(error, axis=1)
    relative = distance/(np.linalg.norm(target.reshape(len(pred),-1),axis=1)+1e-5)
    computed = {"fnorm": distance.mean(), "rmse": np.sqrt(np.mean(error**2)), "mae": np.abs(error).mean(),
                **{f"ewt_{t}": 100*np.mean(relative<t/100) for t in (5,10,25)}}
    for key, value in computed.items():
        if not np.isclose(value, report["test_metrics"][key], rtol=1e-6, atol=1e-6):
            raise ValueError(f"metric mismatch: {key}")
    numeric_history = np.asarray([[v for k,v in row.items()] for row in history], dtype=float)
    if not np.isfinite(numeric_history).all():
        raise ValueError("nonfinite training history")
    timing = report["timing"]
    expected_total = sum(timing[k] for k in ("feature_preparation_seconds", "graph_preparation_seconds", "training_seconds"))
    if not np.isclose(expected_total, timing["total_seconds"]):
        raise ValueError("total timing accounting mismatch")
    for row in timing["inference"]["samples"]:
        if not np.isclose(row["total_seconds"], row["feature_seconds"]+row["graph_seconds"]+row["downstream_seconds"]):
            raise ValueError("inference timing accounting mismatch")
    report["risk"] = {"relative_median": float(np.median(relative)),
                      "relative_p90": float(np.quantile(relative,.90)),
                      "relative_p95": float(np.quantile(relative,.95)),
                      "over_25pct": float(100*np.mean(relative>=.25))}
    report["_distance"], report["_relative"] = distance, relative
    return report


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values)>1 else None


def efficiency_comparisons(reports):
    comparisons={}
    for task in TASKS:
        baseline={r["spec"]["seed"]:r for r in reports if r["spec"]["task"]==task
                  and r["spec"]["model"]=="O(3)" and r["spec"]["fraction"]==100}
        half={r["spec"]["seed"]:r for r in reports if r["spec"]["task"]==task
              and r["spec"]["model"]=="pretrain" and r["spec"]["fraction"]==50}
        if set(baseline)!=set(SEEDS) or set(half)!=set(SEEDS):
            comparisons[task]={"status":"pending three complete seeds for both comparisons"}
            continue
        delta=[half[s]["test_metrics"]["fnorm"]-baseline[s]["test_metrics"]["fnorm"] for s in SEEDS]
        avg,std=mean_std(delta)
        threshold=float(np.mean([r["best_validation_fnorm"] for r in baseline.values()]))
        hits=[]
        for r in reports:
            if r["spec"]["task"]!=task:
                continue
            hit=next((h for h in r["history"] if h["validation_fnorm"]<=threshold),None)
            hits.append({"spec":r["spec"],"epoch":None if hit is None else hit["epoch"],
                         "seconds":None if hit is None else hit["elapsed_training_seconds"],
                         "status":"not reached" if hit is None else "reached"})
        comparisons[task]={"status":"complete", "fnorm_pretrain50_minus_o3_100_mean":avg,
                           "paired_seed_difference_std":std,"point_estimate_matches_or_improves":avg<=0,
                           "interpretation":"descriptive comparison; not a formal noninferiority test",
                           "validation_convergence_threshold":threshold,
                           "threshold_definition":"mean best validation Fnorm of O(3) with 100% labels",
                           "convergence":hits}
    return comparisons


def table(reports):
    columns = [("Fnorm", lambda r:r["test_metrics"]["fnorm"]),
               ("EwT 5% (%)", lambda r:r["test_metrics"]["ewt_5"]),
               ("EwT 10% (%)", lambda r:r["test_metrics"]["ewt_10"]),
               ("EwT 25% (%)", lambda r:r["test_metrics"]["ewt_25"]),
               ("RMSE", lambda r:r["test_metrics"]["rmse"]),
               ("Feature prep (s)", lambda r:r["timing"]["feature_preparation_seconds"]),
               ("Graph prep (s)", lambda r:r["timing"]["graph_preparation_seconds"]),
               ("Training (s)", lambda r:r["timing"]["training_seconds"]),
               ("New structure (ms)", lambda r:1000*r["timing"]["inference"]["mean_seconds_per_structure"]),
               ("Total (s)", lambda r:r["timing"]["total_seconds"])]
    header = ["Task", "Model", "Labels", "Seeds", *[name for name,_ in columns]]
    data = []
    numerical = []
    for task in TASKS:
        for fraction in FRACTIONS:
            for model in LABELS:
                selected = [r for r in reports if r["spec"]["task"] == task
                            and r["spec"]["fraction"] == fraction and r["spec"]["model"] == model]
                cells = [TASKS[task], model, f"{fraction}%", f"{len(selected)}/3"]
                record = {"task": task, "fraction": fraction, "model": model, "seed_count": len(selected)}
                for name, extractor in columns:
                    if selected:
                        avg, std = mean_std([extractor(r) for r in selected])
                        cells.append(f"{avg:.4g}" + (f" ± {std:.3g}" if std is not None else " (one seed)"))
                        record[name] = {"mean": avg, "std": std}
                    else:
                        cells.append("Pending")
                        record[name] = None
                data.append(cells)
                numerical.append(record)
    markup = "<table><thead><tr>"+"".join(f"<th>{html.escape(s)}</th>" for s in header)+"</tr></thead><tbody>"
    markup += "".join("<tr>"+"".join(f"<td>{html.escape(s)}</td>" for s in row)+"</tr>" for row in data)
    return markup+"</tbody></table>", numerical


def curves(reports, task, output, plt):
    fig, axes = plt.subplots(2,4,figsize=(17,7), sharex=True)
    for j, fraction in enumerate(FRACTIONS):
        for model in LABELS:
            selected = [r for r in reports if r["spec"] ["task"]==task
                        and r["spec"]["fraction"]==fraction and r["spec"]["model"]==model]
            if not selected:
                continue
            for i, key in enumerate(("training_loss", "validation_fnorm")):
                values = np.asarray([[r[key] for r in report["history"]] for report in selected])
                x=np.arange(1,values.shape[1]+1)
                avg=values.mean(0)
                axes[i,j].plot(x,avg,label=f"{model} (n={len(selected)})",color=COLORS[model])
                if len(selected)>1:
                    std=values.std(0,ddof=1)
                    axes[i,j].fill_between(x,avg-std,avg+std,color=COLORS[model],alpha=.15)
                axes[i,j].grid(alpha=.2)
        axes[0,j].set_title(f"{fraction}% labels")
        axes[1,j].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Training Huber loss")
    axes[1,0].set_ylabel("Validation Fnorm")
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="upper right",ncol=2)
    fig.suptitle(f"{TASKS[task]} | mean ± training-seed standard deviation")
    fig.tight_layout(rect=(0,0,1,.95))
    save_figure(fig,output,task+"_training_validation")


def grouped(reports, rows, task, fraction, output, plt):
    test=[r for r in rows if r["split"]=="test"]
    train=[r for r in rows if r["split"]=="train"]
    def magnitude(row):
        tensor=np.asarray(row["tensor"])
        if task=="elastic":
            pairs=((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))
            tensor=np.array([[tensor[i,j,k,l] for k,l in pairs] for i,j in pairs])
        return np.linalg.norm(tensor)
    cuts=np.quantile([magnitude(r) for r in train],[.25,.5,.75])
    groupings={"Point group":[r["point_group"] for r in test],
               "Element count":[str(len(set(r["atomic_numbers"]))) for r in test],
               "Structure size":[("1–5" if len(r["atomic_numbers"])<=5 else "6–10" if len(r["atomic_numbers"])<=10
                                   else "11–20" if len(r["atomic_numbers"])<=20 else ">20") for r in test],
               "Target magnitude (training quartiles)":[f"Q{1+np.searchsorted(cuts,magnitude(r),side='right')}" for r in test]}
    fig,axes=plt.subplots(1,4,figsize=(19,9),gridspec_kw={"width_ratios":[2,1,1,1]})
    numerical={"training_magnitude_quartile_edges":cuts.tolist()}
    for ax,(name,assignments) in zip(axes,groupings.items()):
        counts=Counter(assignments)
        groups=sorted(counts,key=lambda g:(-counts[g],g)) if name=="Point group" else sorted(counts)
        y=np.arange(len(groups))
        numerical[name]={}
        for model, offset in (("pretrain",-.18),("O(3)",.18)):
            selected=[r for r in reports if r["spec"]["task"]==task and r["spec"]["model"]==model
                      and r["spec"]["fraction"]==fraction]
            if not selected:
                continue
            values=np.array([[r["_distance"][np.asarray(assignments)==g].mean() for g in groups] for r in selected])
            avg=values.mean(0)
            std=values.std(0,ddof=1) if len(selected)>1 else None
            ax.barh(y+offset,avg,height=.34,xerr=std,color=COLORS[model],label=model,alpha=.9)
            numerical[name][model]={g:{"count":counts[g],"fnorm_mean":float(avg[i]),
                                       "seed_std":None if std is None else float(std[i])} for i,g in enumerate(groups)}
        ax.set_yticks(y,[f"{g} (n={counts[g]})" for g in groups])
        ax.invert_yaxis()
        ax.set(title=name,xlabel="Test Fnorm")
        ax.grid(axis="x",alpha=.2)
    axes[0].legend()
    fig.suptitle(f"{TASKS[task]} | {fraction}% labels | fixed test set | mean ± seed SD")
    fig.tight_layout(rect=(0,0,1,.96))
    name=f"{task}_groups_p{fraction}"
    save_figure(fig,output,name)
    write_json(output/(name+".json"),numerical)


def render(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    plt=setup_plotting()
    reports=[]
    datasets={}
    for task in TASKS:
        rows,provenance=load_records(task,root=ROOT)
        datasets[task]=rows
        dataset_figures(rows,task,output,plt)
        for spec in [s for s in grid() if s["task"]==task]:
            path=args.runs/run_name(spec)/"summary.json"
            if path.is_file():
                reports.append(audit_run(path,spec,provenance,rows))
    markup,numerical=table(reports)
    write_json(output/"metrics.json",numerical)
    status=f"{len(reports)}/48 completed runs"
    body=f"<h1>Pretraining label efficiency</h1><p><strong>{status}</strong>. Missing runs are pending; no estimated metrics.</p>"
    body+="<p>Each row reports mean ± sample standard deviation over training seeds. Separate tasks and units: total dielectric (dimensionless), elastic (GPa). Fixed validation/test sets; nested training subsets.</p>"+markup
    body+="<p>Preparation: measured work allocated to selected train+validation records, including initialization and full serialization overhead. Total = feature preparation + graph preparation + training. Preparation is reused across seeds. New-structure latency sums fresh feature extraction, graph construction and prediction with resident models.</p>"
    comparisons=efficiency_comparisons(reports)
    write_json(output/"efficiency_comparisons.json",comparisons)
    body+="<h2>50% labels versus 100% labels</h2>"
    for task,comparison in comparisons.items():
        if comparison["status"]!="complete":
            body+=f"<p>{TASKS[task]}: pending three seeds for each model.</p>"
        else:
            delta=comparison["fnorm_pretrain50_minus_o3_100_mean"]
            sd=comparison["paired_seed_difference_std"]
            body+=f"<p>{TASKS[task]}: Fnorm difference (pretrain 50% minus O(3) 100%) = {delta:.4g} ± {sd:.3g}. Negative favors pretrain. Descriptive seed comparison; no formal equivalence claim.</p>"
    for task in TASKS:
        body+=f'<h2>{TASKS[task]}</h2><img src="{task}_dataset_distribution.png">'
        task_reports=[r for r in reports if r["spec"]["task"]==task]
        if task_reports:
            curves(reports,task,output,plt)
            body+=f'<img src="{task}_training_validation.png">'
            for fraction in FRACTIONS:
                if any(r["spec"]["fraction"]==fraction for r in task_reports):
                    grouped(reports,datasets[task],task,fraction,output,plt)
                    body+=f'<img src="{task}_groups_p{fraction}.png">'
    page='<!doctype html><meta charset="utf-8"><title>Pretraining experiments</title><style>body{font:15px system-ui;margin:30px;color:#223}table{border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef}img{max-width:100%}p{max-width:1000px;line-height:1.6}</style>'+body
    if "DPA" in page or "GMTNet" in page:
        raise ValueError("forbidden table label")
    (output/"report.html").write_text(page,encoding="utf-8")
    clean=[{k:v for k,v in r.items() if not k.startswith("_")} for r in reports]
    write_json(output/"audit.json",{"status":status,"reports":clean})
    print(status)
    if len(reports)!=48 and not args.allow_partial:
        raise RuntimeError("experiment incomplete; use --allow-partial only for progress reports")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,default=ROOT/"results/pretrain/20261001/runs")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--allow-partial",action="store_true")
    render(parser.parse_args())
