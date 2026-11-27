"""Render recorded training decisions and outer-fold performance using only completed folds."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from elbow import bend


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def curves(directory, title):
    rows = table(directory / "history.csv")
    meta = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    def col(key):
        return np.array([float(r[key]) for r in rows])
    epochs = col("epoch")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    loss, ap, lr, elbow = axes.flat
    # The two losses use different class ratios, so each gets its own axis.
    loss.plot(epochs, col("fit_bce_sampled"), "o-", color="#3b7dd8", label="Fit BCE (resampled 1:neg-ratio)")
    loss.set(ylabel="Fit BCE")
    natural = loss.twinx()
    natural.plot(epochs, col("monitor_bce"), "s-", color="#e07b39", label="Monitor BCE (natural ratio, prior-corrected)")
    natural.set_ylabel("Monitor BCE")
    natural.legend(fontsize=8, loc="upper center")
    ap.plot(epochs, col("monitor_ap"), "o-", color="#156579", label="Monitor AP")
    ap.axhline(col("monitor_pos")[0] / col("monitor_n")[0], color=".45", ls=":", label="Monitor prevalence")
    ap.set(ylabel="Average precision")
    lr.step(epochs, col("lr"), where="post", label="Rate used for epoch", color="#156579")
    lr.set(ylabel="Learning rate", yscale="log")
    distance, idx = bend(epochs, np.minimum.accumulate(col("monitor_bce")))
    elbow.plot(epochs, distance, "o-", color="#825237", label="Running-minimum monitor BCE")
    if idx is not None:
        elbow.scatter(epochs[idx], distance[idx], s=95, facecolors="none", edgecolors="#ad4e27",
                      label=f"Descriptive bend at epoch {int(epochs[idx])}")
    elbow.set(ylabel="Normalized distance from chord")
    for ax in (loss, ap):
        ax.axvline(meta["best_epoch"], ls="--", color="#ad4e27", label="Restored checkpoint")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title}  |  {directory.name}  |  inner Train monitor\n"
                 f"Selected by {meta['monitor'].upper()} at epoch {meta['best_epoch']}    ·    "
                 f"Ended at {meta['end_epoch']} ({meta['stop_reason']})")
    fig.savefig(directory / "learning.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    conf = json.loads((run / "run.json").read_text(encoding="utf-8"))["args"]
    title = f"BEV {conf['resolution']} · {conf['frames']} frame(s) · {conf['channels']} · {conf['shards']}/40 Train shards"
    predictions, fold_labels = [], []
    dirs = sorted(run.glob("fold-*/summary.json"))
    if not dirs:
        parser.error("No completed folds")
    for file in dirs:
        curves(file.parent, title)
        fold_predictions = table(file.parent / "predictions.csv")
        predictions.extend(fold_predictions)
        fold_labels.extend([file.parent.name] * len(fold_predictions))
    if len({r["sample_id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate outer-fold predictions")
    y = np.array([int(r["label"]) for r in predictions])
    s = np.array([float(r["score"]) for r in predictions])
    hit = s >= np.array([float(r["threshold"]) for r in predictions])
    lead = np.array([float(r["lead_time_s"]) if r["lead_time_s"] else np.nan for r in predictions])
    ap = average_precision_score(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    pr, cm, by_time = axes
    pr.step(recall, precision, where="post", color="#156579", label=f"Pooled AP {ap:.3f}")
    pr.axhline(y.mean(), ls=":", color="#ad4e27", label=f"Prevalence {y.mean():.3f}")
    tp = np.sum(hit & (y == 1))
    pr.scatter(tp/y.sum(), tp/max(1,hit.sum()), color="#ad4e27", label="Inner-fold threshold(s)")
    pr.set(xlabel="Recall", ylabel="Precision", xlim=(-.02,1.02), ylim=(-.02,1.02), title="Precision–recall")
    pr.legend(fontsize=8)
    counts = np.array([[np.sum((y==v) & (hit==h)) for h in (0,1)] for v in (0,1)])
    ratio = counts / counts.sum(axis=1, keepdims=True)
    cm.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    for (i,j), v in np.ndenumerate(counts):
        cm.text(j,i,f"{v:,}\n{ratio[i,j]:.1%}",ha="center",va="center",
                color="white" if ratio[i,j]>.55 else "#182b3b")
    cm.set(xticks=[0,1], yticks=[0,1], xlabel="Predicted label", ylabel="True label", title="Counts and row percentages")
    bins = []
    for lo in (0,1,2):
        valid = (y==1) & (lead>lo+.005) & (lead<=lo+1+.005)
        found, total = int(hit[valid].sum()), int(valid.sum())
        value = found / total if total else None
        bins.append(dict(lower_s=lo, upper_s=lo+1, found=found, positives=total, recall=value))
        if total:
            by_time.bar(lo,value,color="#156579",width=.6)
        by_time.text(lo, (value or 0)+.03, f"{found}/{total}" if total else "n/a", ha="center")
    by_time.set(xticks=range(3), xticklabels=["(0, 1]", "(1, 2]", "(2, 3]"], ylim=(0,1.15),
                ylabel="Recall", xlabel="Seconds until entry", title="Recall by lead time")
    missing = int(np.sum((y==1)&~np.isfinite(lead)))
    fig.suptitle(f"{title}  |  {len(dirs)}/5 outer folds\n"
                 f"n = {len(y):,}    ·    positives = {y.sum():,}    ·    positives missing lead time = {missing}")
    fig.savefig(run / "evaluation.png", dpi=200)
    plt.close(fig)
    fold_labels = np.array(fold_labels)
    by_fold = {name: float(average_precision_score(y[fold_labels==name], s[fold_labels==name]))
               for name in sorted(set(fold_labels))}
    with (run / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(dict(folds=list(by_fold), outer_n=len(y), positives=int(y.sum()), pooled_ap=float(ap),
                       fold_ap=by_fold, mean_fold_ap=float(np.mean(list(by_fold.values()))),
                       confusion=counts.tolist(), lead_bins=bins, positives_missing_time=missing), f, indent=2)


if __name__ == "__main__":
    main()

