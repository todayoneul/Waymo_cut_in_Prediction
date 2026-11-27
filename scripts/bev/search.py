"""Randomized hyperparameter search on fold 1's inner monitor split (Lec2 p.62, Lec10 pp.57-59).

Each trial runs train.py with --fold 1 --tune, so the fold-1 outer segments are never scored.
Trials are ranked by the monitor AP of the restored checkpoint.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def numbers(text, kind):
    return [kind(v) for v in text.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", type=int, default=40)
    p.add_argument("--resolution", choices=["1m", "20cm"], default="1m")
    p.add_argument("--frames", choices=[1, 11], type=int, default=1)
    p.add_argument("--channels", choices=["all", "pair", "pair-map"], default="all")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--lr-min", type=float, required=True, help="Usually suggested_lr / 30 from lr_find")
    p.add_argument("--lr-max", type=float, required=True, help="Usually suggested_lr from lr_find")
    p.add_argument("--widths", default="16,32,48")
    p.add_argument("--dropouts", default="0,0.2,0.4")
    p.add_argument("--weight-decays", default="0,1e-5,1e-4,1e-3")
    p.add_argument("--batches", default="32,64")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    if not 0 < args.lr_min <= args.lr_max or args.trials < 1:
        p.error("Need 0 < lr-min <= lr-max and at least one trial")
    repo = args.repo.resolve()
    data = args.data or repo / "data/bev_sequences"
    space = dict(width=numbers(args.widths, int), dropout=numbers(args.dropouts, float),
                 weight_decay=numbers(args.weight_decays, float), batch=numbers(args.batches, int))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "space.json").write_text(json.dumps(dict(lr=[args.lr_min, args.lr_max], **space,
                                                         args={k: str(v) for k, v in vars(args).items()}),
                                                    indent=2), encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    trials = []
    fields = ["trial", "lr", "width", "dropout", "weight_decay", "batch", "status",
              "best_monitor_ap", "best_epoch", "end_epoch", "stop_reason"]
    for number in range(1, args.trials + 1):
        config = dict(lr=float(np.exp(rng.uniform(np.log(args.lr_min), np.log(args.lr_max)))),
                      **{name: rng.choice(values).item() for name, values in space.items()})
        folder = args.out / f"trial-{number:02d}"
        command = [sys.executable, str(Path(__file__).with_name("train.py")), "--repo", str(repo),
                   "--data", str(data), "--out", str(folder), "--fold", "1", "--tune",
                   "--shards", str(args.shards), "--resolution", args.resolution, "--frames", str(args.frames),
                   "--channels", args.channels, "--epochs", str(args.epochs), "--patience", str(args.patience),
                   "--lr-patience", str(args.lr_patience), "--workers", str(args.workers),
                   "--device", args.device, "--seed", str(args.seed), "--lr", f"{config['lr']:.6g}",
                   "--width", str(config["width"]), "--dropout", str(config["dropout"]),
                   "--weight-decay", str(config["weight_decay"]), "--batch", str(config["batch"])]
        print(f"trial {number}/{args.trials}: {config}", flush=True)
        finished = subprocess.run(command).returncode == 0
        record = dict(trial=number, **config, status="ok" if finished else "failed",
                      best_monitor_ap=float("nan"), best_epoch="", end_epoch="", stop_reason="")
        if finished:
            summary = json.loads((folder / "fold-1/summary.json").read_text(encoding="utf-8"))
            record.update({k: summary[k] for k in ("best_monitor_ap", "best_epoch", "end_epoch", "stop_reason")})
        trials.append(record)
        with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(trials)

    done = [t for t in trials if t["status"] == "ok"]
    if not done:
        raise SystemExit("Every trial failed. Lower --lr-max or check the training log.")
    best = max(done, key=lambda t: t["best_monitor_ap"])
    (args.out / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    ap = np.array([t["best_monitor_ap"] for t in done])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    for ax, key in zip(axes, ("lr", "width", "dropout", "weight_decay")):
        values = np.array([t[key] for t in done], dtype=float)
        points = ax.scatter(values, ap, c=[t["batch"] for t in done], cmap="viridis", s=45)
        ax.scatter(best[key], best["best_monitor_ap"], s=160, facecolors="none", edgecolors="#ad4e27", lw=2)
        ax.set(xlabel=key, ylabel="Fold-1 monitor AP (restored checkpoint)")
        ax.grid(alpha=.2)
    axes[0].set_xscale("log")
    axes[3].set_xscale("symlog", linthresh=1e-5)
    fig.colorbar(points, ax=axes[-1], label="Batch size")
    fig.suptitle(f"Randomized search  |  BEV {args.resolution} · {args.frames} frame(s) · {args.channels}  |  "
                 f"{len(done)}/{len(trials)} finished trials   ·   color = batch size   ·   circle = best")
    fig.savefig(args.out / "search.png", dpi=200)
    plt.close(fig)
    print("best:", json.dumps(best))


if __name__ == "__main__":
    main()
