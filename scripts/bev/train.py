"""Train one BEV configuration on Train-only grouped folds. Run from the repository root."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

CHANNELS = ["lane_center", "road_lines_and_edges", "reference", "target", "other_vehicles"]
# The two map layers do not change over time, so they enter the network once.
# Vehicle layers (reference, target, other vehicles) are stacked for every frame, oldest first.
MAP_LAYERS = {"all": [0, 1], "pair-map": [0, 1], "pair": []}
VEHICLE_LAYERS = {"all": [0, 1, 2], "pair-map": [0, 1], "pair": [0, 1]}


def load_repo_metrics(repo):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.dataset import evaluate, validation_threshold
    return evaluate, validation_threshold


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Sequences(Dataset):
    def __init__(self, directory, count, resolution, frames, channels, mirror=False):
        self.mirror = mirror
        self.rows = []
        self.files = []
        self.handles = {}
        self.resolution = resolution
        self.frames = frames
        self.map_layers = np.array(MAP_LAYERS[channels], dtype=int)
        self.vehicle_layers = np.array(VEHICLE_LAYERS[channels], dtype=int)
        self.in_channels = len(self.map_layers) + frames * len(self.vehicle_layers)
        self.digests = {}
        for number in range(count):
            path = directory / f"train_{number:05d}.zip"
            with ZipFile(path) as archive:
                raw = archive.read("manifest.json")
                manifest = json.loads(raw)
                if manifest["excerpt"] or manifest["channels"] != CHANNELS:
                    raise ValueError(f"Unexpected BEV schema in {path}")
                np.testing.assert_allclose(manifest["offsets_s"], np.arange(-10, 1) / 10)
                if manifest["resolutions"][resolution] != {"1m": 1, "20cm": .2}[resolution]:
                    raise ValueError(f"Unexpected resolution in {path}")
                if not manifest["samples"] or any(r["split"] != "train" or str(r["label"]) not in {"0", "1"}
                                                  for r in manifest["samples"]):
                    raise ValueError(f"Invalid Train manifest in {path}")
                self.digests[path.name] = hashlib.sha256(raw).hexdigest()
                self.rows.extend(manifest["samples"])
                self.files.extend([path] * len(manifest["samples"]))
        if len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate Train sample_id")
        self.y = np.array([int(r["label"]) for r in self.rows])
        self.groups = np.array([r["segment_id"] for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path = self.files[idx]
        if path not in self.handles:
            self.handles[path] = ZipFile(path)
        row = self.rows[idx]
        with np.load(io.BytesIO(self.handles[path].read(row["member"])), allow_pickle=False) as arrays:
            maps = np.unpackbits(arrays["map_" + self.resolution][self.map_layers], axis=-1)
            vehicles = np.unpackbits(arrays["cars_" + self.resolution][-self.frames:, self.vehicle_layers], axis=-1)
            target_on_right = self.mirror and arrays["centers"][-1, 1, 1] < 0
        planes = np.concatenate((maps, vehicles.reshape(-1, *vehicles.shape[-2:])))
        if target_on_right:
            # Mirror about the reference vehicle's row (row r <-> H - r) so the target is always on the left (+y),
            # the same convention src/features.py uses for the tabular features.
            planes = np.roll(planes[:, ::-1], 1, axis=1)
        return torch.from_numpy(planes), int(row["label"])

    def __getstate__(self):
        return {**self.__dict__, "handles": {}}


class EpochSampler(Sampler):
    """Index list that can change between passes, so one DataLoader keeps its worker processes."""

    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loader(data, batch, workers, device):
    return DataLoader(data, batch_size=batch, sampler=EpochSampler(), num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=device == "cuda")


def activation(name):
    # Lec11 pp.12-15: leaky ReLU keeps a small gradient for negative inputs, so units cannot die.
    return nn.LeakyReLU(0.01) if name == "leaky" else nn.ReLU()


def conv_block(in_channels, out_channels, kernel, stride, padding, batchnorm, act="relu"):
    # Lec11 p.23: BN before the activation makes the convolution bias redundant.
    layers = [nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=not batchnorm)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation(act))
    return layers


def network(in_channels, resolution, width, dropout, batchnorm, act="relu"):
    """Conv -> [BatchNorm] -> ReLU blocks, a 3x6 spatial grid and a small dense head.

    The 1 m input (48x96) passes three 3x3 stride-2 blocks and ends at 6x12 (receptive field 15 m).
    The 20 cm input (240x480) first passes a 5x5 stride-5 block that turns each 1 m cell's 25 sub-cells
    into learned features, then the same three blocks. Both resolutions therefore share the grid and
    receptive field, and a comparison isolates the effect of sub-metre detail.
    """
    layers = []
    if resolution == "20cm":
        layers += conv_block(in_channels, width, 5, 5, 0, batchnorm, act)
        in_channels = width
    for out_channels in (width, 2 * width, 4 * width):
        layers += conv_block(in_channels, out_channels, 3, 2, 1, batchnorm, act)
        in_channels = out_channels
    head = [nn.Linear(in_channels * 18, 64, bias=not batchnorm)]
    if batchnorm:
        head.append(nn.BatchNorm1d(64))
    head.extend([activation(act), nn.Dropout(dropout), nn.Linear(64, 1)])
    model = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten(), *head)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # He initialization, Lec11 pp.9-11
            nn.init.kaiming_normal_(module.weight, a=0.01 if act == "leaky" else 0, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    nn.init.xavier_uniform_(model[-1].weight)  # sigmoid output layer
    return model


def fold_partitions(y, groups, seed):
    """Five outer folds by drive segment. Each outer training part is split again into fit and monitor segments."""
    for fold, (fit_pool, outer) in enumerate(GroupKFold(5).split(y, y, groups), 1):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed + fold)
        fit_idx, monitor_idx = next(inner.split(fit_pool, y[fit_pool], groups[fit_pool]))
        yield fold, fit_pool[fit_idx], fit_pool[monitor_idx], outer


def logit(p):
    return math.log(p / (1 - p))


@torch.inference_mode()
def predict(model, loader, indices, device, prior_shift):
    """Scores at the natural class ratio. prior_shift undoes the 1:neg-ratio resampling used in training."""
    model.eval()
    loader.sampler.indices = indices.tolist()
    logits = []
    labels = []
    for images, y in loader:
        logits.append(model(images.to(device, dtype=torch.float32, non_blocking=True)).flatten().cpu())
        labels.append(y)
    logits = torch.cat(logits) + prior_shift
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.cat(labels).float()).item()
    if not math.isfinite(loss):
        raise ValueError("Nonfinite evaluation loss")
    return logits.sigmoid().numpy(), loss


def run_fold(args, data, loader, fold, fit, monitor, outer, device):
    out = args.out / f"fold-{fold}"
    out.mkdir()
    y, groups = data.y, data.groups
    seed = args.seed + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    partitions = {"fit": fit, "monitor": monitor} if args.tune else {"fit": fit, "monitor": monitor, "outer": outer}
    with (out / "groups.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["partition", "segment_id"])
        for name, subset in partitions.items():
            if len(np.unique(y[subset])) != 2:
                raise ValueError(f"{name} lacks both classes. Include more Train shards.")
            w.writerows((name, g) for g in np.unique(groups[subset]))

    model = network(data.in_channels, args.resolution, args.width, args.dropout, not args.no_batchnorm,
                    args.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    maximize = args.monitor == "ap"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max" if maximize else "min", patience=args.lr_patience, factor=.5,
        threshold=args.min_delta, threshold_mode="abs")
    pos, neg = fit[y[fit] == 1], fit[y[fit] == 0]
    n_negatives = min(len(neg), args.neg_ratio * len(pos))
    prior_shift = logit(y[fit].mean()) - logit(len(pos) / (len(pos) + n_negatives))
    best = reference = -np.inf
    best_epoch = stale = 0
    fields = ["epoch", "fit_bce_sampled", "monitor_bce", "monitor_ap", "lr", "lr_next",
              "fit_n_sampled", "monitor_n", "monitor_pos", "stale_epochs", "elapsed_s"]
    checkpoint = out / "best.pt"
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            sampled = rng.permutation(np.r_[pos, rng.choice(neg, n_negatives, replace=False)])
            if not args.no_batchnorm and len(sampled) > 1 and len(sampled) % args.batch == 1:
                sampled = sampled[:-1]
            loader.sampler.indices = sampled.tolist()
            model.train()
            total = 0.
            lr = optimizer.param_groups[0]["lr"]
            for images, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to(device, dtype=torch.float32, non_blocking=True)).flatten()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device).float())
                if not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                if args.clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(labels)
            scores, bce = predict(model, loader, monitor, device, prior_shift)
            ap = float(average_precision_score(y[monitor], scores))
            metric = ap if maximize else bce
            gain = metric if maximize else -metric
            if gain > best:
                best, best_epoch = gain, epoch
                torch.save(model.state_dict(), out / "pending.pt")
                os.replace(out / "pending.pt", checkpoint)
            if gain > reference + args.min_delta:
                reference, stale = gain, 0
            else:
                stale += 1
            if args.schedule == "plateau":
                scheduler.step(metric)
            w.writerow(dict(zip(fields, [epoch, total / len(sampled), bce, ap, lr,
                       optimizer.param_groups[0]["lr"], len(sampled), len(monitor),
                       int(y[monitor].sum()), stale, time.monotonic() - start])))
            fp.flush()
            print(f"fold {fold} epoch {epoch}: monitor AP {ap:.4f}, BCE {bce:.4f}, "
                  f"{time.monotonic() - start:.0f}s", flush=True)
            if args.patience and stale >= args.patience:
                break
    summary = dict(best_epoch=best_epoch, end_epoch=epoch, monitor=args.monitor,
                   best_monitor_value=float(abs(best)), prior_shift_logit=prior_shift,
                   stop_reason="patience" if args.patience and stale >= args.patience else "epoch_limit",
                   fit_n=len(fit), monitor_n=len(monitor), outer_n=None if args.tune else len(outer))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    inner_scores, _ = predict(model, loader, monitor, device, prior_shift)
    summary["best_monitor_ap"] = float(average_precision_score(y[monitor], inner_scores))
    if args.tune:
        save_json(out / "summary.json", summary)
        return
    evaluate, validation_threshold = load_repo_metrics(args.repo)
    threshold = validation_threshold(y[monitor], inner_scores)
    scores, _ = predict(model, loader, outer, device, prior_shift)
    summary["metrics"] = evaluate(y[outer], scores, threshold)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["sample_id", "segment_id", "label", "lead_time_s", "score", "threshold"])
        for idx, score in zip(outer, scores):
            row = data.rows[idx]
            w.writerow([row["sample_id"], row["segment_id"], y[idx], row["time_target_s"],
                        float(score), threshold])
    save_json(out / "summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="Waymo_cut_in_Prediction repository root")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", type=int, default=40, choices=range(1, 41), metavar="1..40")
    p.add_argument("--fold", type=int, default=0, choices=range(6), help="1..5 or 0 for all")
    p.add_argument("--tune", action="store_true", help="Inner monitor only. Skips the outer fold (used by search.py)")
    p.add_argument("--resolution", choices=["1m", "20cm"], default="1m")
    p.add_argument("--frames", choices=[1, 11], type=int, default=1)
    p.add_argument("--channels", choices=["all", "pair", "pair-map"], default="all")
    p.add_argument("--width", type=int, default=32, help="Channels of the first 1 m block. Later blocks use 2x and 4x")
    p.add_argument("--no-batchnorm", action="store_true", help="Remove BatchNorm2d (ablation)")
    p.add_argument("--activation", choices=["relu", "leaky"], default="relu")
    p.add_argument("--mirror", action="store_true",
                   help="Flip rasters so the target is always on the reference vehicle's left, as in src/features.py")
    p.add_argument("--epochs", type=int, default=100, help="Upper bound. Early stopping usually ends earlier")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--neg-ratio", type=int, default=4)
    p.add_argument("--monitor", choices=["ap", "bce"], default="ap")
    p.add_argument("--patience", type=int, default=8, help="0 = run all epochs, restore best for evaluation")
    p.add_argument("--min-delta", type=float, default=.001)
    p.add_argument("--schedule", choices=["plateau", "constant"], default="plateau")
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--clip", type=float, default=0.)
    p.add_argument("--dropout", type=float, default=0.)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    args.repo = args.repo.resolve()
    if args.data is None:
        args.data = args.repo / "data/bev_sequences"
    load_repo_metrics(args.repo)
    if (min(args.epochs, args.batch, args.neg_ratio, args.width) < 1 or args.lr <= 0 or not 0 <= args.dropout < 1
            or min(args.min_delta, args.patience, args.lr_patience, args.workers, args.clip, args.weight_decay) < 0):
        p.error("Invalid training parameters")
    if not args.no_batchnorm and args.batch < 2:
        p.error("BatchNorm requires --batch >= 2")
    if args.tune and args.fold == 0:
        p.error("--tune needs a single --fold")
    data = Sequences(args.data, args.shards, args.resolution, args.frames, args.channels, args.mirror)
    device = pick_device(args.device)
    model = network(data.in_channels, args.resolution, args.width, args.dropout, not args.no_batchnorm, args.activation)
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    save_json(args.out / "run.json", dict(started=datetime.now(timezone.utc).isoformat(),
              args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              in_channels=data.in_channels, parameters=sum(t.numel() for t in model.parameters()),
              architecture=str(model),
              git=commit.stdout.strip(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              platform=platform.platform(), python=sys.version, torch=torch.__version__, numpy=np.__version__,
              sklearn=sklearn.__version__, cuda=torch.version.cuda, device=device,
              gpu=torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
              manifests_sha256=data.digests))
    loader = make_loader(data, args.batch, args.workers, device)
    for fold, fit, monitor, outer in fold_partitions(data.y, data.groups, args.seed):
        if args.fold in (0, fold):
            run_fold(args, data, loader, fold, fit, monitor, outer, device)


if __name__ == "__main__":
    main()
