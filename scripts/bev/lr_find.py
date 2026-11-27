"""Learning-rate range test on fold 1's fit segments (Lec10 p.63, Geron ch.10).

Trains for a few hundred mini-batches while the learning rate grows exponentially, then reports
the rate where the smoothed loss is lowest and the largest rate before the loss diverges.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from train import Sequences, fold_partitions, make_loader, network, pick_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", type=int, default=40, choices=range(1, 41), metavar="1..40")
    p.add_argument("--resolution", choices=["1m", "20cm"], default="1m")
    p.add_argument("--frames", choices=[1, 11], type=int, default=1)
    p.add_argument("--channels", choices=["all", "pair", "pair-map"], default="all")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--no-batchnorm", action="store_true")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--neg-ratio", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr-start", type=float, default=1e-6)
    p.add_argument("--lr-end", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    if args.steps < 20 or not 0 < args.lr_start < args.lr_end:
        p.error("Need at least 20 steps and 0 < lr-start < lr-end")
    data_dir = args.data or args.repo.resolve() / "data/bev_sequences"
    data = Sequences(data_dir, args.shards, args.resolution, args.frames, args.channels)
    device = pick_device(args.device)
    _, fit, _, _ = next(fold_partitions(data.y, data.groups, args.seed))
    pos, neg = fit[data.y[fit] == 1], fit[data.y[fit] == 0]
    rng = np.random.default_rng(args.seed + 1)
    order = []
    while len(order) < args.steps * args.batch:
        negatives = rng.choice(neg, min(len(neg), args.neg_ratio * len(pos)), replace=False)
        order.extend(rng.permutation(np.r_[pos, negatives]).tolist())
    loader = make_loader(data, args.batch, args.workers, device)
    loader.sampler.indices = order[:args.steps * args.batch]

    torch.manual_seed(args.seed + 1)
    model = network(data.in_channels, args.resolution, args.width, 0., not args.no_batchnorm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0.)
    growth = (args.lr_end / args.lr_start) ** (1 / (args.steps - 1))
    records = []
    average = 0.
    lowest = math.inf
    diverged = False
    model.train()
    for step, (images, labels) in enumerate(loader):
        rate = args.lr_start * growth ** step
        for group in optimizer.param_groups:
            group["lr"] = rate
        logits = model(images.to(device, dtype=torch.float32)).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device).float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = loss.item()
        average = 0.98 * average + 0.02 * value
        smoothed = average / (1 - 0.98 ** (step + 1))
        records.append((step, rate, value, smoothed))
        lowest = min(lowest, smoothed)
        if not math.isfinite(value) or smoothed > 4 * lowest:
            diverged = True
            break

    rates = np.array([r[1] for r in records])
    smooth = np.array([r[3] for r in records])
    usable = np.isfinite(smooth)
    turning = float(rates[usable][np.argmin(smooth[usable])])
    max_stable = float(rates[-2]) if diverged and len(rates) > 1 else None
    # Lec10 p.63: the optimal rate is about half the maximum stable rate. Geron ch.10 also suggests
    # about 10x below the turning point. The smaller (safer) of the two becomes the suggestion.
    candidates = [turning / 10] + ([max_stable / 2] if max_stable else [])
    suggestion = min(candidates)

    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "lr_find.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "loss", "smoothed_loss"])
        w.writerows(records)
    result = dict(turning_point_lr=turning, max_stable_lr=max_stable, suggested_lr=suggestion,
                  diverged=diverged, steps_run=len(records), batch=args.batch, resolution=args.resolution,
                  frames=args.frames, channels=args.channels, width=args.width,
                  batchnorm=not args.no_batchnorm, shards=args.shards, fit_n=len(fit), device=device)
    (args.out / "lr_find.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.plot(rates, [r[2] for r in records], color=".75", lw=.8, label="Mini-batch loss")
    ax.plot(rates, smooth, color="#156579", lw=2, label="Smoothed loss (EMA 0.98)")
    ax.axvline(turning, color="#825237", ls="--", label=f"Lowest smoothed loss  {turning:.2e}")
    if max_stable:
        ax.axvline(max_stable, color="#ad4e27", ls=":", label=f"Last rate before divergence  {max_stable:.2e}")
    ax.axvline(suggestion, color="#2d7a3e", lw=2, label=f"Suggested learning rate  {suggestion:.2e}")
    ax.set(xscale="log", xlabel="Learning rate (log scale)", ylabel="Training BCE on resampled batches",
           ylim=(0, min(max(smooth[usable]) * 1.1, 3 * smooth[usable][0])))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    ax.set_title(f"Learning-rate range test  |  BEV {args.resolution} · {args.frames} frame(s) · {args.channels} · "
                 f"width {args.width} · batch {args.batch}\n{len(records)} steps on fold-1 fit segments "
                 f"({args.shards}/40 Train shards)")
    fig.savefig(args.out / "lr_find.png", dpi=200)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
