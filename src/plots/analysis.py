"""데이터 분석 그림(train만 사용): 라벨 구성, feature 분포·상관, PCA 비교."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEGATIVE, POSITIVE = "#247588", "#d05d34"
INK, MUTED, GRID = "#182a3c", "#4b6273", "#dce5ea"
SELECTED = {
    "fwd": "Forward gap (m)",
    "lat": "Lateral gap (m)",
    "dvy": "Relative lateral speed (m/s)",
    "lat_rate_10": "1 s lateral gap change (m/s)",
    "own_off": "Target offset from lane center (m)",
    "ref_lead_dist": "Gap to lead vehicle (m)",
}
SOURCE_LABELS = ("Rule label", "Human label")


def configure_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "#ffffff", "legend.frameon": False,
    })


def save(fig, out, name):
    fig.savefig(Path(out) / f"{name}.png", dpi=160)
    plt.close(fig)


def title(fig, main, sub):
    fig.suptitle(f"{main}\n{sub}", linespacing=1.55, fontsize=14, fontweight="bold")


def plot_label_overview(counts, output):
    splits = ("train", "validation", "test")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), layout="constrained")
    title(fig, "How the dataset is labeled", "Split by source file · label balance shown for train only")
    total = [counts[s]["samples"] for s in splits]
    bars = axes[0].barh(splits, total, color=NEGATIVE, height=.45)
    train_pos = counts["train"]["positives"]
    axes[0].barh(["train"], [train_pos], color=POSITIVE, height=.45, label="Label 1 (train)")
    axes[0].bar_label(bars, labels=[f"{n:,}" + (f"  ·  label 1 {100 * train_pos / n:.1f}%" if s == "train" else "")
                                    for s, n in zip(splits, total)], padding=5, fontsize=9)
    axes[0].set(xlim=(0, max(total) * 1.6), xlabel="Samples", title="Samples per split")
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[0].legend(loc="lower right", fontsize=8)
    rule = [counts[s]["label_sources"].get("rule", 0) for s in splits]
    human = [counts[s]["label_sources"].get("human", 0) for s in splits]
    y = np.arange(len(splits))
    for values, offset, color, label in zip((rule, human), (-.16, .16), (NEGATIVE, "#7c5caf"), SOURCE_LABELS):
        bars = axes[1].barh(y + offset, values, height=.26, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    axes[1].set(yticks=y, yticklabels=splits, xscale="log", xlim=(.8, max(rule) * 2.7),
                xlabel="Label count (log scale)", title="Label source")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.30), ncol=2, fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, alpha=.6)
        ax.set_axisbelow(True)
    save(fig, output, "01_label_overview")


def plot_feature_distributions(x, y, names, output):
    indexes = [names.index(n) for n in SELECTED]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.1), layout="constrained")
    title(fig, "Feature distributions", "Training only · normalized within each class")
    for j, label, ax in zip(indexes, SELECTED.values(), axes.flat):
        col = x[:, j]
        lo, hi = np.nanpercentile(col, [1, 99])
        if lo == hi:
            lo, hi = lo - .5, hi + .5
        bins = np.linspace(lo, hi, 23)
        for c, color, tag in ((0, NEGATIVE, "Label 0"), (1, POSITIVE, "Label 1")):
            data = col[(y == c) & np.isfinite(col)]
            if len(data):
                ax.hist(np.clip(data, lo, hi), bins=bins, weights=np.full(len(data), 100 / len(data)),
                        histtype="step", linewidth=1.8, color=color, label=tag)
        ax.set(title=label, ylabel="Within class (%)")
        if label == SELECTED["ref_lead_dist"]:
            ax.set_xlabel("80 m includes no lead vehicle detected", fontsize=8)
        ax.grid(axis="y", color=GRID, alpha=.7)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    save(fig, output, "02_feature_distributions")

    chosen = x[:, indexes]
    medians = np.nanmedian(chosen, axis=0)
    filled = np.where(np.isnan(chosen), medians, chosen)
    corr = np.corrcoef(filled, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.5), layout="constrained")
    title(fig, "Feature correlations", "Training only · Pearson r after median imputation")
    tags = ("Forward gap", "Lateral gap", "Lateral speed", "1 s gap change", "Lane offset", "Lead vehicle gap")
    pic = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tags)), tags, rotation=28, ha="right")
    ax.set_yticks(range(len(tags)), tags)
    for (i, j), value in np.ndenumerate(corr):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > .58 else INK)
    fig.colorbar(pic, ax=ax, shrink=.84, label="Pearson r")
    save(fig, output, "03_correlations")


def plot_pca(pcs, variance, labels, results, output):
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.1), layout="constrained")
    title(fig, "Does dimensionality reduction help?", "Training data only · 5-fold cross-validation by drive segment")
    cumulative = np.cumsum(variance)
    axes[0].plot(np.arange(1, len(variance) + 1), cumulative * 100, color=NEGATIVE, linewidth=2)
    for fraction in (.90, .95):
        n = int(np.searchsorted(cumulative, fraction) + 1)
        axes[0].scatter(n, fraction * 100, color=POSITIVE, zorder=3)
        axes[0].annotate(f"{n} PCs at {fraction:.0%}", (n, fraction * 100), xytext=(8, -13),
                         textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Components", ylabel="Explained variance (%)", title="Compression")
    for c, color, tag in ((0, NEGATIVE, "Label 0"), (1, POSITIVE, "Label 1")):
        mask = labels == c
        axes[1].scatter(pcs[mask, 0], pcs[mask, 1], s=8, c=color, alpha=.18 if c == 0 else .38,
                        edgecolors="none", label=tag)
    axes[1].set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})", title="Training samples in 2D")
    axes[1].legend(loc="upper right", fontsize=8)
    names = [f"{name} ({m['columns']})" for name, m in results.items()]
    bars = axes[2].barh(names, [m["average_precision"] for m in results.values()],
                         height=.5, color=[NEGATIVE, "#769ba2", "#47868d", "#ab8bb9", "#9365aa"])
    axes[2].set(title="Train cross-validation", xlabel="Average precision", xlim=(0, 1))
    axes[2].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[2].invert_yaxis()
    for ax in (axes[0], axes[2]):
        ax.grid(color=GRID, alpha=.6)
        ax.set_axisbelow(True)
    save(fig, output, "04_pca_comparison")
