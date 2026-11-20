"""성능 평가 그림. train 주행 구간 5-fold 교차검증의 예측값(OOF)만 사용하고 validation·test는 쓰지 않는다."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from sklearn.metrics import average_precision_score, precision_recall_curve

from .scene import ROAD, legend_below, outline, road_axes
from .style import (BLUE_DARK, BLUE_LIGHT, INK, INPUT_NAMES, MODEL_COLORS, MODEL_NAMES, MUTED, NEG, OTHER, REF, TGT,
                    grid_lines, save)

BEST = "boosting_balanced__selected_76"   # train 교차검증 AP가 가장 높은 조합
LEAD_BINS = (0, .505, 1.005, 1.505, 2.005, 2.505, 9)  # 0.5초 구간. 10 Hz 시각 오차를 흡수하도록 경계를 0.005초 늦춘다.
CURVES = (  # (예측값 이름, 그림 이름, 선 모양)
    ("logistic__selected_76", "Logistic Regression · Selected 76", "-"),
    ("random_forest__selected_76", "Random Forest · Selected 76", "-"),
    ("boosting_balanced__selected_76", "Gradient Boosting (cw) · Selected 76", "-"),
    ("boosting_balanced__motion_64", "Gradient Boosting (cw) · Motion 64", (0, (4, 2))),
    ("boosting_balanced__basic_8", "Gradient Boosting (cw) · Basic 8", (0, (1, 1.5))),
)


def plot_model_comparison(metrics, output):
    models, inputs = list(MODEL_NAMES), list(INPUT_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    width = .16
    panels = (("average_precision", "AP (Average Precision, PR curve 아래 넓이)"), ("f1", "F1 (F1이 가장 높은 Threshold 기준)"))
    for ax, (key, title) in zip(axes, panels):
        values = np.array([[metrics[f"{m} · {i}"]["train_cv_oof"][key] for i in inputs] for m in models])
        for row, model in enumerate(models):
            ax.bar(np.arange(len(inputs)) + (row - 2) * width, values[row], width=width * .92,
                   color=MODEL_COLORS[model], label=MODEL_NAMES[model])
        for col in range(len(inputs)):
            best = int(values[:, col].argmax())
            ax.text(col + (best - 2) * width, values[best, col] + .015, f"{values[best, col]:.3f}", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold")
        if key == "average_precision":
            rate = metrics[f"{models[0]} · {inputs[0]}"]["train_cv_oof"]["positive_rate"]
            ax.axhline(rate, color=MUTED, lw=1, ls=(0, (3, 2)))
            baseline = Line2D([0], [0], color=MUTED, lw=1, ls=(0, (3, 2)), label=f"Random baseline (AP {rate:.3f})")
        ax.set_xticks(range(len(inputs)), [INPUT_NAMES[i] for i in inputs])
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, loc="left")
        grid_lines(ax)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + [baseline], loc="upper center", bbox_to_anchor=(.5, 0), ncol=3, fontsize=10.5)
    save(fig, output, "01_model_comparison")


def best_threshold(y, scores):
    """F1이 가장 높은 임계값 (같으면 높은 값). src.dataset.validation_threshold와 같은 규칙."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]]), thresholds, p, r, f1


def plot_pr_and_threshold(y, oof, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    for key, name, dash in CURVES:
        precision, recall, _ = precision_recall_curve(y, oof[key])
        color = MODEL_COLORS[key.split("__")[0]]
        ax.plot(recall, precision, color=color, ls=dash, lw=2.6 if key == BEST else 1.8,
                label=f"{name}  AP {average_precision_score(y, oof[key]):.3f}")
    threshold, _, _, _, _ = best_threshold(y, oof[BEST])
    predicted = oof[BEST] >= threshold
    point = ((predicted & (y == 1)).sum() / (y == 1).sum(), (predicted & (y == 1)).sum() / predicted.sum())
    ax.scatter(*point, s=90, color=TGT, edgecolor="white", lw=1.2, zorder=5)
    ax.annotate(f"F1 최대 Threshold 지점\nPrecision {point[1]:.2f} · Recall {point[0]:.2f}", point, xytext=(.99, .99),
                textcoords="data", ha="right", va="top", fontsize=10.5, color=TGT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TGT, lw=1))
    ax.axhline(y.mean(), color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (실제 Cut-in 중 찾은 비율)", ylabel="Precision (예측한 Cut-in 중 맞은 비율)")
    ax.set_title("Precision-Recall curve", loc="left")
    ax.legend(loc="lower left", fontsize=10, handlelength=2.6)
    grid_lines(ax, "both")

    ax = axes[1]
    threshold, thresholds, precision, recall, f1 = best_threshold(y, oof[BEST])
    for values, color, name in ((precision, BLUE_DARK, "Precision"), (recall, TGT, "Recall"), (f1, INK, "F1")):
        ax.plot(thresholds, values, color=color, lw=2.2 if name == "F1" else 1.8, ls="-" if name != "F1" else (0, (4, 2)),
                label=name)
    ax.axvline(threshold, color=MUTED, lw=1)
    ax.text(threshold - .015, .05, f"F1 최대 Threshold {threshold:.2f}\nF1 {f1.max():.3f}", fontsize=10.5, va="bottom",
            ha="right")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Threshold (예측 확률이 이 값 이상이면 Cut-in으로 판정)")
    ax.set_title("Threshold에 따른 Precision · Recall · F1 (Gradient Boosting · Selected 76)", loc="left", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(.02, .66), fontsize=10.5)
    grid_lines(ax, "both")
    save(fig, output, "02_pr_and_threshold")
    return threshold


def plot_error_breakdown(y, scores, rows, threshold, output):
    predicted = scores >= threshold
    fig = plt.figure(figsize=(12, 4.2), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[.85, 1.35, .95])

    ax = fig.add_subplot(grid[0])
    cells = np.array([[(~predicted & (y == 0)).sum(), (predicted & (y == 0)).sum()],
                      [(~predicted & (y == 1)).sum(), (predicted & (y == 1)).sum()]])
    colors = np.array([[to_rgb("#eef1f4"), to_rgb("#fde3d2")], [to_rgb("#fde3d2"), to_rgb("#cfe0f5")]])
    ax.imshow(colors)
    names = [["TN", "FP (잘못 경고)"], ["FN (놓친 Cut-in)", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - .08, f"{cells[i, j]:,}", ha="center", va="center", fontsize=15, fontweight="bold")
            ax.text(j, i + .22, names[i][j], ha="center", va="center", fontsize=10.5, color=MUTED)
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Confusion matrix", loc="left")

    ax = fig.add_subplot(grid[1])
    lead = np.array([float(r["lead_time_s"]) if r["lead_time_s"] else np.nan for r in rows])
    found, total = [], []
    for low, high in zip(LEAD_BINS, LEAD_BINS[1:]):
        pick = (y == 1) & (lead > low) & (lead <= high)
        found.append(int((pick & predicted).sum()))
        total.append(int(pick.sum()))
    recall = np.array(found) / np.array(total)
    labels = ["0~0.5", "0.5~1", "1~1.5", "1.5~2", "2~2.5", "2.5~3"]
    bars = ax.bar(labels, recall, color=[TGT if r < .6 else BLUE_DARK for r in recall], width=.65)
    ax.bar_label(bars, labels=[f"{r:.0%}\n{f}/{t}" for r, f, t in zip(recall, found, total)], padding=3, fontsize=10)
    ax.set(ylim=(0, 1.18), xlabel="Lead time (s)")
    ax.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
    ax.set_title("Lead time별 Recall", loc="left")
    grid_lines(ax)

    ax = fig.add_subplot(grid[2])
    kinds = (("waymo_sdc", "Waymo SDC"), ("other_vehicle", "Other vehicle"))
    values = []
    for key, _ in kinds:
        mask = np.array([r["reference_type"] == key for r in rows])
        values.append(float(average_precision_score(y[mask], scores[mask])))
    bars = ax.barh([name for _, name in kinds], values, color=[REF, BLUE_LIGHT], height=.55)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Reference 종류별 AP", loc="left")
    save(fig, output, "03_error_breakdown")
    return dict(zip(labels, recall.tolist()))


def draw_input_scene(ax, bev, sequence):
    """모델이 보는 입력을 도로 장면처럼 그린다.

    BEV의 주변 차량 칸은 회색으로 칠하고, 1 m 격자의 선 채널(Lane centerline, Road line·Road edge)은 칸 중심에 점으로 찍는다.
    두 차량은 Sequence의 과거 1초 경로와 현재 위치·크기로 그린다.
    """
    image = np.tile(np.array(to_rgb(ROAD)), (48, 96, 1))
    image[bev[4].astype(bool)] = to_rgb(OTHER)
    ax.imshow(image, extent=(-24.5, 71.5, -23.5, 24.5), interpolation="nearest", zorder=0)
    for channel, color, size in ((0, "#61778b", 2), (1, "#d5dee6", 4)):
        rows, cols = np.nonzero(bev[channel])
        ax.scatter(cols - 24, 24 - rows, s=size, marker="s", color=color, linewidths=0, zorder=1)
    for vehicle, color in ((0, REF), (1, TGT)):
        x, y, vx, vy, yaw, length, width = sequence[-1, vehicle]
        state = (x, y, yaw, vx, vy, length, width, 1)
        ax.plot(sequence[:, vehicle, 0], sequence[:, vehicle, 1], color=color, lw=2.6, zorder=4)
        ax.add_patch(Polygon(outline(state, lambda p: p), closed=True, facecolor=color, edgecolor="white", lw=1.2, zorder=5))
    road_axes(ax, (-22, 62), (-11, 13))


def plot_error_examples(examples, output):
    """examples = [(제목, bev, sequence)] 네 개."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 4.9), gridspec_kw=dict(hspace=.28, wspace=.04))
    for ax, (title, bev, sequence) in zip(axes.flat, examples):
        draw_input_scene(ax, bev, sequence)
        ax.set_title(title, loc="left", fontsize=12.5)
    handles = [Patch(facecolor=REF, edgecolor="white", label="Reference"), Patch(facecolor=TGT, edgecolor="white", label="Target"),
               Patch(facecolor=OTHER, label="Other vehicle"), Line2D([0], [0], color="white", lw=2.6, label="Past 1 s trajectory"),
               Line2D([0], [0], color="#d5dee6", ls=(0, (1, 1.5)), lw=2.5, label="Road line · Road edge (BEV 1 m grid)"),
               Line2D([0], [0], color="#61778b", ls=(0, (1, 2)), lw=2, label="Lane centerline")]
    legend_below(fig, handles, ncol=6, y=.04)
    save(fig, output, "04_error_examples")

