"""Explore driving contexts using Train-only standardized basic features, K-means and silhouette."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

def _load_basic_features(repo):
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from src.dataset import BASIC_FEATURES
    return BASIC_FEATURES


def bend(x, y):
    """Vertical distance below the endpoint chord after range normalization."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x[-1] <= x[0] or y[0] <= y[-1]:
        return np.zeros_like(x), None
    distance = (y[0] - y) / (y[0] - y[-1]) - (x - x[0]) / (x[-1] - x[0])
    i = int(np.argmax(distance))
    return distance, i if 0 < i < len(x)-1 and distance[i] > 1e-8 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="Waymo_cut_in_Prediction repository root")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--max-k", type=int, default=12)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    args.repo = args.repo.resolve()
    if args.dataset is None:
        args.dataset = args.repo / "data/dataset"
    BASIC_FEATURES = _load_basic_features(args.repo)
    with (args.dataset / "samples.csv").open(encoding="utf-8-sig") as f:
        ids = sorted(r["sample_id"] for r in csv.DictReader(f) if r["split"] == "train")
    rng = np.random.default_rng(args.seed)
    n = min(len(ids), args.n)
    if not 3 <= args.max_k < min(n, 1500):
        p.error("Require 3 <= max-k < min(Train sample size, 1500)")
    selected = set(rng.choice(ids, n, replace=False))
    values, sample_ids = [], []
    with (args.dataset / "features_basic.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["sample_id"] in selected:
                values.append([float(r[k]) for k in BASIC_FEATURES])
                sample_ids.append(r["sample_id"])
    if len(values) != n or set(sample_ids) != selected:
        raise ValueError("Train feature alignment failed")
    scale = StandardScaler()
    x = scale.fit_transform(values)
    subset = rng.choice(n, min(1500, n), replace=False)
    args.out.mkdir(parents=True, exist_ok=False)
    ks = np.arange(1, args.max_k + 1)
    inertias, silhouettes = [], []
    with (args.out / "kmeans_trials.csv").open("w", encoding="utf-8", newline="") as f, threadpool_limits(limits=2):
        w = csv.writer(f)
        w.writerow(["k", "seed", "n_init", "train_n", "silhouette_n", "inertia", "silhouette"])
        for k in ks:
            inertia_k, silhouette_k = [], []
            for offset in (0, 1, 2):
                model = KMeans(n_clusters=int(k), n_init=5, random_state=args.seed + offset).fit(x)
                labels = model.labels_[subset]
                score = silhouette_score(x[subset], labels) if 1 < len(set(labels)) < len(labels) else np.nan
                w.writerow([k, args.seed + offset, 5, n, len(subset), model.inertia_,
                            score if np.isfinite(score) else ""])
                inertia_k.append(model.inertia_)
                silhouette_k.append(score)
            inertias.append(inertia_k)
            silhouettes.append(silhouette_k)
    inertia = np.array(inertias)
    sil = np.array(silhouettes)
    med = np.median(inertia, axis=1)
    distance, idx = bend(ks, med)
    candidate = None if idx is None else int(ks[idx])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, values_, title in [(axes[0], inertia, "Within-cluster sum of squares"),
                               (axes[2], sil, "Mean silhouette")]:
        median = np.median(values_, axis=1)
        ax.plot(ks, median, "o-", color="#156579")
        ax.fill_between(ks, values_.min(axis=1), values_.max(axis=1), color="#156579", alpha=.17)
        ax.set_title(title)
    axes[1].plot(ks, distance, "o-", color="#825237")
    axes[1].axhline(0, lw=.8, color="0.5")
    axes[1].set_title("Normalized bend from endpoint chord")
    for ax in axes:
        if candidate:
            ax.axvline(candidate, color="#ad4e27", ls="--", label=f"Candidate k = {candidate}")
        ax.set(xlabel="Number of clusters (k)", xticks=ks)
        ax.grid(axis="y", alpha=.2)
    if candidate:
        axes[1].legend()
    fig.suptitle(f"Train driving contexts  |  {n:,} samples, 8 standardized features\n"
                 f"Median and range across 3 seeds   ·   Silhouette on {len(subset):,} fixed samples")
    fig.savefig(args.out / "elbow.png", dpi=200)
    plt.close(fig)
    with (args.out / "elbow_curve.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "median_inertia", "drop_from_previous_k", "normalized_bend", "median_silhouette"])
        for i,k in enumerate(ks):
            w.writerow([k, med[i], med[i-1]-med[i] if i else "", distance[i],
                        np.median(sil[i]) if np.isfinite(sil[i]).all() else ""])
    (args.out / "samples.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps(dict(seed=args.seed, features=BASIC_FEATURES,
        scaler_mean=scale.mean_.tolist(), scaler_scale=scale.scale_.tolist(), n=n, max_k=args.max_k,
        candidate=candidate, silhouette_n=len(subset)), indent=2), encoding="utf-8")
    print(f"Train n={n}, elbow candidate k={candidate}")


if __name__ == "__main__":
    main()

