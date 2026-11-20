"""데이터 분석 그림 (train만 사용): 라벨 구성, feature 분포·상관, feature 선택, PCA 비교."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import BLUE_DARK, BLUE_LIGHT, BLUE_MID, GRID, INK, MUTED, NEG, REF, TGT, grid_lines, save

# 그림에 쓰는 feature 설명. 이름은 src/features.py와 같다.
FEATURE_LABELS = {
    "fwd": "앞뒤 거리", "lat": "좌우 거리", "dvx": "앞뒤 상대속도", "dvy": "좌우 상대속도",
    "time_gap": "앞뒤 time gap", "lat_rate_10": "1 s 동안 좌우 간격 변화율",
    "fwd_pred_2": "2 s 뒤 앞뒤 거리 (등속 가정)", "lat_pred_1": "1 s 뒤 좌우 거리 (등속 가정)",
    "lat_pred_fit_2": "2 s 뒤 좌우 거리 (2차 추세)", "lat_pred_fit_1": "1 s 뒤 좌우 거리 (2차 추세)",
    "own_off": "자기 lane 중심에서 벗어난 정도", "d_ref_lane": "Reference lane 중심까지 좌우 거리",
    "d_ref_lane_rate_5": "Reference lane 접근 속도 (0.5 s)", "time_to_ref_lane": "Reference lane 도달 예상 시간",
    "ref_lead_minus_tgt": "Reference 앞 공간 - Target 위치", "tgt_lat_vel_lane": "lane 기준 좌우 속도",
    "tgt_lead_headway": "Target 앞차 time headway", "tgt_yaw_rate_2": "yaw rate (0.2 s)",
    "tgt_yaw_rate_5": "yaw rate (0.5 s)", "tgt_yaw_rate_10": "yaw rate (1 s)",
    "inner_line_dist": "안쪽 road line까지 거리", "outer_line_dist": "바깥쪽 road line까지 거리",
    "yaw_quad": "yaw 변화의 곡률", "yaw_slope_fit": "yaw 변화의 기울기", "tgt_heading_lane": "lane 대비 heading",
    "lane_sep_ahead_40": "40 m 앞 두 lane 간격", "ref_lead_dist": "Reference 앞차 거리",
    "lat_speed_from_heading": "heading으로 계산한 좌우 속도", "v_ref": "Reference 속력", "v_tgt": "Target 속력",
}
DISTRIBUTIONS = {  # feature: 단위
    "fwd": "m", "lat": "m", "dvy": "m/s", "own_off": "m", "d_ref_lane": "m", "tgt_lead_headway": "s (20 s = 앞차 없음)",
}
PCA_NAMES = {"Full": "All 99", "Imputed control": "Standardized 99 (대조군)", "Correlation filter": "Correlation filter",
             "PCA 90%": "PCA (variance 90%)", "PCA 95%": "PCA (variance 95%)"}
LEAD_EDGES = (1.005, 2.005)  # 진입까지 남은 시간 구간 경계. 10 Hz 시각 오차(±0.001초)를 흡수하도록 베이스라인 코드와 같은 값을 쓴다.


def label_of(name):
    """feature 코드 이름과 뜻을 함께 쓴다. 예: fwd_pred_2 · 2 s 뒤 앞뒤 거리 (등속 가정)"""
    return f"{name} · {FEATURE_LABELS[name]}" if name in FEATURE_LABELS else name


def plot_class_balance(y, rows, output):
    fig = plt.figure(figsize=(12, 3.9), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[.85, 1.3, 1])

    ax = fig.add_subplot(grid[0])
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars = ax.bar(["No cut-in (0)", "Cut-in (1)"], counts, color=[NEG, TGT], width=.6)
    ax.bar_label(bars, labels=[f"{counts[0]:,}", f"{counts[1]:,}\n({counts[1] / sum(counts):.2%})"], padding=3, fontsize=11)
    ax.set_ylim(0, counts[0] * 1.25)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("train Label 분포", loc="left")

    ax = fig.add_subplot(grid[1])
    lead = np.array([float(r["lead_time_s"]) for r, label in zip(rows, y) if label == 1])
    heights, _, _ = ax.hist(lead, bins=np.arange(-.05, 3.06, .1), color=TGT, edgecolor="white", lw=.6)  # 0.1초 시점마다 막대 하나
    ax.set_ylim(0, heights.max() * 1.22)
    edges = (0, *LEAD_EDGES, 9)
    for i, (low, high) in enumerate(zip(edges, edges[1:])):
        if i:
            ax.axvline(low, color=INK, lw=.8, ls=(0, (2, 2)))
        share = ((lead > low) & (lead <= high)).mean()
        ax.text(i + .5, heights.max() * 1.2, f"{i}~{i + 1} s  {share:.1%}", ha="center", va="top", fontsize=11.5,
                fontweight="bold")
    ax.set(xlim=(0, 3), xlabel="Lead time: 예측 시점부터 Entry까지 (s)", ylabel="Sample 수")
    ax.set_title(f"Cut-in {len(lead):,}건의 Lead time (median {np.median(lead):.1f} s)", loc="left")
    grid_lines(ax)

    ax = fig.add_subplot(grid[2])
    kinds = (("waymo_sdc", "Waymo SDC"), ("other_vehicle", "Other vehicle"))
    totals, rates = [], []
    for key, _ in kinds:
        mask = np.array([r["reference_type"] == key for r in rows])
        totals.append(int(mask.sum()))
        rates.append(float(y[mask].mean()))
    bars = ax.barh([name for _, name in kinds], [100 * r for r in rates], color=[REF, BLUE_MID], height=.55)
    ax.bar_label(bars, labels=[f"{100 * r:.2f}%  (n = {n:,})" for r, n in zip(rates, totals)], padding=4, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(rates) * 100 * 1.9)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Reference 종류별 Positive 비율", loc="left")
    save(fig, output, "01_class_balance")


def plot_feature_distributions(x, y, names, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), layout="constrained")
    for ax, (name, unit) in zip(axes.flat, DISTRIBUTIONS.items()):
        column = x[:, names.index(name)]
        # 두 라벨 각각의 1~99% 범위를 모두 담는다. 범위 밖 값(라벨마다 2% 이하)은 양 끝 막대에 모인다.
        ranges = [np.nanpercentile(column[y == label], [1, 99]) for label in (0, 1)]
        low, high = min(r[0] for r in ranges), max(r[1] for r in ranges)
        bins = np.linspace(low, high, 31)
        for label, color, tag in ((0, NEG, "No cut-in (0)"), (1, TGT, "Cut-in (1)")):
            values = column[(y == label) & np.isfinite(column)]
            ax.hist(np.clip(values, low, high), bins=bins, weights=np.full(len(values), 100 / len(values)),
                    histtype="stepfilled" if label == 0 else "step", color=color, alpha=.35 if label == 0 else 1,
                    lw=2.2, label=tag)
        ax.set_title(label_of(name), loc="left", fontsize=12.5)
        ax.set_xlabel(unit)
        grid_lines(ax)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes.flat[0].legend(loc="upper right", fontsize=10.5)
    fig.supylabel("Label별 비율 (density)", fontsize=11.5, color=MUTED)
    save(fig, output, "02_feature_distributions")


def plot_feature_correlations(x, names, ranked, output, count=12):
    chosen = ranked[:count]
    columns = x[:, [names.index(n) for n in chosen]]
    filled = np.where(np.isnan(columns), np.nanmedian(columns, axis=0), columns)
    corr = np.corrcoef(filled, rowvar=False)
    fig, ax = plt.subplots(figsize=(10.5, 6.6), layout="constrained")
    picture = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [label_of(n) for n in chosen]
    ax.set_xticks(range(count), range(1, count + 1))
    ax.set_yticks(range(count), [f"{i}. {t}" for i, t in enumerate(labels, 1)], fontsize=10.5)
    ax.tick_params(length=0)
    for (i, j), value in np.ndenumerate(corr):
        if i != j:
            ax.text(j, i, f"{value:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > .6 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(picture, ax=ax, shrink=.8, pad=.02)
    bar.outline.set_visible(False)
    bar.set_label("Pearson r")
    fig.suptitle(f"Permutation importance 상위 {count}개 feature의 Pearson correlation", x=.02, ha="left",
                 fontsize=13.5, fontweight="bold")
    save(fig, output, "03_feature_correlations")


def plot_feature_selection(selection, motion_names, output, count=15):
    importance = selection["importance_after_correlation_filter"]
    ranked = sorted(importance, key=importance.get, reverse=True)
    fig = plt.figure(figsize=(12, 5.2), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])

    ax = fig.add_subplot(grid[0])
    top = ranked[:count]
    values = [importance[n] for n in top]
    colors = [BLUE_MID if n in motion_names else TGT for n in top]
    bars = ax.barh([label_of(n) for n in top], values, color=colors, height=.65)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.15)
    ax.tick_params(axis="y", labelsize=10.5, length=0)
    ax.set_xlabel("feature 값을 섞었을 때 CV AP 감소량")
    ax.set_title(f"Permutation importance 상위 {count}개", loc="left")
    ax.legend(handles=[Patch(color=BLUE_MID, label="Motion feature"), Patch(color=TGT, label="Road · Interaction feature")],
              loc="lower right", fontsize=10.5)
    grid_lines(ax, "x")

    ax = fig.add_subplot(grid[1])
    folds = selection["train_cv_average_precision"]
    n_all = len(selection["importance_after_correlation_filter"]) + len(selection["dropped_by_correlation"])
    settings = [("all", f"All {n_all}"),
                ("correlation_filtered", f"Correlation filter {len(importance)}"),
                ("selected", f"Importance 선택 {len(selection['selected'])}")]
    for i, (key, name) in enumerate(settings):
        values = np.array(folds[key])
        ax.scatter(values, np.full(len(values), i), color=BLUE_LIGHT, s=46, zorder=3)
        ax.scatter(values.mean(), i, color=BLUE_DARK, s=120, marker="|", lw=3, zorder=4)
        ax.text(values.mean(), i - .28, f"mean {values.mean():.3f}", ha="center", fontsize=11)
    ax.set_yticks(range(3), [name for _, name in settings])
    ax.set_ylim(2.6, -.7)
    ax.set_xlabel("train 5-fold CV AP")
    ax.set_title("Feature 수에 따른 성능 (점 = fold)", loc="left")
    grid_lines(ax, "x")
    save(fig, output, "04_feature_selection")
    return ranked


def plot_pca(pcs, variance, labels, results, output):
    fig = plt.figure(figsize=(12, 4.2), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    cumulative = np.cumsum(variance)

    ax = fig.add_subplot(grid[0])
    ax.plot(np.arange(1, len(variance) + 1), cumulative * 100, color=BLUE_DARK, lw=2.4)
    for fraction in (.90, .95):
        n = int(np.searchsorted(cumulative, fraction) + 1)
        ax.scatter(n, fraction * 100, color=TGT, s=40, zorder=3)
        ax.annotate(f"{fraction:.0%}: PC {n}개", (n, fraction * 100), xytext=(8, -16), textcoords="offset points",
                    fontsize=10.5)
    ax.set(xlabel="Principal component 수", ylabel="Cumulative explained variance (%)", ylim=(0, 102))
    ax.set_title("PCA explained variance", loc="left")
    grid_lines(ax, "both")

    ax = fig.add_subplot(grid[1])
    for label, color in ((0, NEG), (1, TGT)):
        mask = labels == label
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=5, c=color, alpha=.12 if label == 0 else .45, edgecolors="none")
    low, high = np.percentile(pcs[:, :2], [.5, 99.5], axis=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel=f"PC1 ({variance[0]:.0%})", ylabel=f"PC2 ({variance[1]:.0%})")
    ax.set_title("train Sample의 PC1-PC2 분포", loc="left")
    dot = lambda color, tag: Line2D([0], [0], marker="o", ls="none", markerfacecolor=color, markeredgecolor="none",
                                    markersize=8, label=tag)
    ax.legend(handles=[dot(NEG, "No cut-in"), dot(TGT, "Cut-in")], loc="upper right", fontsize=10.5)

    ax = fig.add_subplot(grid[2])
    names = [PCA_NAMES[k] for k in results]
    values = [m["average_precision"] for m in results.values()]
    colors = [BLUE_DARK, BLUE_MID, BLUE_MID, "#c3ccd5", "#c3ccd5"]
    bars = ax.barh(names, values, color=colors, height=.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("train 5-fold CV AP (Gradient Boosting)")
    ax.set_title("Input 표현별 성능", loc="left")
    grid_lines(ax, "x")
    save(fig, output, "05_pca")

