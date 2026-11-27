"""Measure BEV loading speed for several DataLoader worker counts and estimate one epoch's loading time."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from train import Sequences, fold_partitions, make_loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="CSV file")
    p.add_argument("--shards", type=int, default=40, choices=range(1, 41), metavar="1..40")
    p.add_argument("--resolution", choices=["1m", "20cm"], default="1m")
    p.add_argument("--frames", choices=[1, 11], type=int, default=1)
    p.add_argument("--channels", choices=["all", "pair", "pair-map"], default="all")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--neg-ratio", type=int, default=4)
    p.add_argument("--workers", default="0,2,4,8")
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    data = Sequences(args.data or args.repo.resolve() / "data/bev_sequences", args.shards,
                     args.resolution, args.frames, args.channels)
    _, fit, monitor, _ = next(fold_partitions(data.y, data.groups, args.seed))
    positives = int(data.y[fit].sum())
    per_epoch = positives + min(len(fit) - positives, args.neg_ratio * positives) + len(monitor)
    picks = np.random.default_rng(args.seed).choice(len(data), min(args.samples, len(data)), replace=False)
    rows = []
    for workers in [int(v) for v in args.workers.split(",")]:
        start = time.perf_counter()
        loader = make_loader(data, args.batch, workers, "cpu")
        loader.sampler.indices = picks.tolist()
        batches = iter(loader)
        first_images, first_labels = next(batches)
        startup = time.perf_counter() - start
        count = 0
        timer = time.perf_counter()
        for images, labels in batches:
            count += len(labels)
        rate = count / (time.perf_counter() - timer)
        rows.append(dict(workers=workers, startup_s=round(startup, 2), samples_per_s=round(rate, 1),
                         epoch_loading_s=round(per_epoch / rate, 1), samples_per_epoch=per_epoch,
                         input_shape="x".join(map(str, first_images.shape[1:]))))
        print(rows[-1], flush=True)
        del batches, loader
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fastest = max(rows, key=lambda r: r["samples_per_s"])
    print(f"Fastest: --workers {fastest['workers']} ({fastest['samples_per_s']} samples/s, "
          f"about {fastest['epoch_loading_s']} s of loading per epoch)")


if __name__ == "__main__":
    main()
