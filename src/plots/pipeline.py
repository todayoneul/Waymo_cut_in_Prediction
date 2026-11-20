"""원본 장면에서 학습 표본이 만들어지고 전체 데이터셋이 완성되는 과정을 그린다."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from ..raw_scenario import frame_of
from .scene import LABEL_SHADOW, ROAD, draw_agents, draw_road, legend_below, outline, road_axes, road_legend, view_bounds
from .style import (BLUE_DARK, BLUE_LIGHT, BLUE_MID, CYCLIST, GRID, INK, MUTED, NEG, OTHER, PEDESTRIAN, REF, TGT,
                    grid_lines, save)

HISTORY, HORIZON = 10, 30               # 10 Hz 기준 과거 1초(입력), 미래 3초(정답 확인)
TYPE_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}
TYPE_COLORS = {1: OTHER, 2: PEDESTRIAN, 3: CYCLIST}
MAP_NAMES = {"lane": "Lane centerline", "road_line": "Road line", "road_edge": "Road edge",
             "crosswalk": "Crosswalk", "speed_bump": "Speed bump", "driveway": "Driveway", "stop_sign": "Stop sign"}


def relative_track(reference, target, frames):
    """각 시점마다 그 시점의 기준 차량 좌표계에서 본 상대 차량의 앞뒤/좌우 위치."""
    values = []
    for j in frames:
        transform, _ = frame_of(reference[j])
        values.append(transform([target[j, :2]])[0])
    return np.array(values)


def maneuver_bounds(tracks, row):
    """기준 시점의 기준 차량 좌표계에서 두 차량의 -1초~+3초 경로가 모두 보이는 범위."""
    ref, tgt = tracks[int(row["reference_id"])], tracks[int(row["target_id"])]
    t0 = int(row["anchor_frame"])
    transform, _ = frame_of(ref[t0])
    frames = np.arange(t0 - HISTORY, min(t0 + HORIZON, len(ref) - 1) + 1)
    return view_bounds(np.vstack([transform(ref[frames, :2]), transform(tgt[frames, :2])]))


def draw_maneuver(ax, items, scene, tracks, row, bounds):
    """기준 시점(0초)의 Reference와 Target만 그리고 과거 1초(실선), 미래 3초(점선), 진입 확인 위치(점선 상자)를 겹친다.

    주변 차량은 두 차량의 움직임을 가려서 이 그림에서는 그리지 않는다.
    """
    ref_id, tgt_id = int(row["reference_id"]), int(row["target_id"])
    ref, tgt = tracks[ref_id], tracks[tgt_id]
    t0, entry = int(row["anchor_frame"]), int(row["entry_frame"])
    transform, _ = frame_of(ref[t0])
    draw_road(ax, items, transform)
    draw_agents(ax, scene, tracks, t0, transform, roles={ref_id: (REF, "Reference"), tgt_id: (TGT, "Target")},
                show_people=False, show_others=False)
    last = min(t0 + HORIZON, len(ref) - 1)
    for states, color in ((ref, REF), (tgt, TGT)):
        past = transform(states[t0 - HISTORY:t0 + 1, :2])
        future = transform(states[t0:last + 1, :2])
        ax.plot(*past.T, color=color, lw=3, zorder=9)
        ax.plot(*future.T, color=color, lw=1.8, ls=(0, (3, 2.5)), zorder=9)
        ax.add_patch(Polygon(outline(states[entry], transform), closed=True, facecolor="none", edgecolor=color,
                             lw=1.6, ls=(0, (3, 2)), zorder=9))
    star = transform(np.array([tgt[entry, :2]]))[0]
    ax.scatter(*star, marker="*", s=260, color="#ffe08a", edgecolor=ROAD, lw=.8, zorder=11)
    ax.annotate(f"Entry +{(entry - t0) / 10:.1f} s", star, xytext=(0, 14), textcoords="offset points",
                color="#ffe08a", fontsize=11.5, fontweight="bold", ha="center", va="bottom", zorder=11,
                path_effects=LABEL_SHADOW)
    road_axes(ax, *bounds)


def maneuver_legend():
    return [Line2D([0], [0], color="white", lw=3, label="Past 1 s (모델 Input)"),
            Line2D([0], [0], color="white", lw=1.8, ls=(0, (3, 2.5)), label="Future 3 s (Label 생성에만 사용)"),
            Patch(facecolor="none", edgecolor="white", ls=(0, (3, 2)), label="Entry 시점의 위치")]


def plot_cutin_example(output_dir, scene, items, tracks, row):
    bounds = maneuver_bounds(tracks, row)
    width = 12.0
    fig = plt.figure(figsize=(width, width * np.ptp(bounds[1]) / np.ptp(bounds[0])))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_maneuver(ax, items, scene, tracks, row, bounds)
    legend_below(ax, road_legend()[:2] + maneuver_legend(), ncol=5)
    save(fig, output_dir, "01_cutin_example")


def plot_raw_scene(output_dir, scene, items, tracks, sdc_id):
    valid_xy = np.vstack([s[s[:, 7] > 0, :2] for s in tracks.values() if (s[:, 7] > 0).any()])
    low, high = valid_xy.min(axis=0) - 10, valid_xy.max(axis=0) + 10
    current = scene.current_time_index
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    draw_road(ax, items)
    for track in scene.tracks:
        states = tracks[track.id]
        ok = states[:, 7] > 0
        if track.id == sdc_id:
            ax.plot(states[ok, 0], states[ok, 1], color=REF, lw=2.6, zorder=7)
        else:
            ax.plot(states[ok, 0], states[ok, 1], color=TYPE_COLORS.get(track.object_type, OTHER), lw=1, alpha=.55, zorder=4)
    draw_agents(ax, scene, tracks, current, lambda xy: np.asarray(xy, dtype=float), roles={sdc_id: (REF, "Waymo SDC")})
    road_axes(ax, (low[0], high[0]), (low[1], high[1]), show_ticks=True, scale_bar=False)
    ax.tick_params(labelsize=10, length=0)
    ax.set_xlabel("Global x (m)", fontsize=11)
    ax.set_ylabel("Global y (m)", fontsize=11)
    handles = [Line2D([0], [0], color="white", lw=1.4, label="Track 경로 (0~9 s)"),
               Patch(facecolor=OTHER, edgecolor="#c9d4de", label="Current time 위치 (1.0 s)")]
    legend_below(ax, handles + road_legend(include_people=True)[3:5], ncol=4, y=-.14)
    save(fig, output_dir, "02_raw_scene")


def plot_raw_contents(output_dir, scene, tracks):
    fig = plt.figure(figsize=(5.6, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax = fig.add_subplot(grid[0, 0])
    kinds = [1, 2, 3]
    counts = [sum(t.object_type == k for t in scene.tracks) for k in kinds]
    bars = ax.barh([TYPE_NAMES[k] for k in kinds], counts, color=[TYPE_COLORS[k] for k in kinds], height=.6)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.3)
    ax.set_title("Track (object type)", loc="left")
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])

    ax = fig.add_subplot(grid[0, 1])
    names = ["lane", "road_line", "road_edge", "crosswalk", "driveway", "speed_bump"]
    counts = [sum(f.WhichOneof("feature_data") == n for f in scene.map_features) for n in names]
    bars = ax.barh([MAP_NAMES[n] for n in names], counts, color=BLUE_MID, height=.6)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.35)
    ax.set_title("Map feature", loc="left")
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])

    ax = fig.add_subplot(grid[1, :])
    ordered = sorted(scene.tracks, key=lambda t: (t.object_type, int(np.argmax(tracks[t.id][:, 7] > 0))))
    matrix = np.array([np.where(tracks[t.id][:, 7] > 0, t.object_type, 0) for t in ordered])
    ax.imshow(matrix, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#f1f3f5", OTHER, PEDESTRIAN, CYCLIST]), vmin=0, vmax=3)
    current = scene.current_time_index
    ax.axvline(current, color=INK, lw=1.4)
    ax.set_xticks(np.arange(0, 91, 10), [f"{t:.0f}" for t in np.arange(0, 9.1, 1)])
    ax.set_yticks([])
    ax.set_title(f"Track {len(ordered)}개 × 91 frames의 valid 여부 (세로선 = current time)", loc="left")
    ax.set_xlabel("Scenario time (s)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, output_dir, "03_raw_contents")


def plot_candidate_pairs(output_dir, scene, items, tracks, row):
    ref_id, tgt_id = int(row["reference_id"]), int(row["target_id"])
    t0 = int(row["anchor_frame"])
    transform, _ = frame_of(tracks[ref_id][t0])
    xlim, ylim = (-22, 62), (-13, 15.5)
    fig = plt.figure(figsize=(6.4, 6.4 * np.ptp(ylim) / np.ptp(xlim)))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_road(ax, items, transform)
    for side in (1, -1):  # 상대 차량 후보 영역: 앞뒤 -15~55 m, 좌우 2.3~11 m
        ax.add_patch(Rectangle((-15, 2.3 if side > 0 else -11), 70, 8.7, facecolor="none", edgecolor="#ffe08a",
                               lw=1.5, ls=(0, (4, 3)), zorder=5))
    ax.text(-14.5, 11.6, "Target 후보 영역: 앞뒤 -15~55 m, 좌우 2.3~11 m", color="#ffe08a", fontsize=10.5,
            va="bottom", zorder=12, path_effects=LABEL_SHADOW)
    draw_agents(ax, scene, tracks, t0, transform, roles={ref_id: (REF, "Reference"), tgt_id: (TGT, "Target")}, show_people=False)
    road_axes(ax, xlim, ylim, scale_bar=False)  # 영역 크기를 글자로 적었으므로 축척 막대는 생략
    legend_below(ax, road_legend()[:3], ncol=3)
    save(fig, output_dir, "04_candidate_pairs")


def plot_anchor_timeline(output_dir, samples):
    positives = [s for s in samples if s["label"] == "1"]
    negatives = [s for s in samples if s["label"] != "1"]
    picked = positives + negatives[::max(1, len(negatives) // 5)][:5]
    picked.sort(key=lambda s: (int(s["anchor_frame"]), s["label"] != "1"))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for t in range(10, 61, 10):
        ax.axvline(t / 10, color=GRID, lw=1, zorder=0)
    for y, sample in enumerate(picked):
        t = int(sample["anchor_frame"]) / 10
        ax.barh(y, 1.0, left=t - 1.0, height=.5, color=REF)
        ax.barh(y, 3.0, left=t, height=.5, color=TGT, alpha=.28)
        if sample["entry_frame"]:
            ax.scatter(int(sample["entry_frame"]) / 10, y, marker="*", s=170, color=TGT, edgecolor=INK, lw=.6, zorder=4)
        who = "Waymo" if sample["reference_type"] == "waymo_sdc" else f"#{sample['reference_id']}"
        ax.text(-.12, y, f"{who} → #{sample['target_id']}", ha="right", va="center", fontsize=10.5)
        verdict, color = ("Label 1", TGT) if sample["label"] == "1" else ("Label 0", MUTED)
        ax.text(t + 3.1, y, verdict, va="center", fontsize=10.5, color=color, fontweight="bold")
    ax.set(xlim=(0, 9), ylim=(len(picked) - .4, -.6), yticks=[], xlabel="Scenario time (s)")
    ax.spines["left"].set_visible(False)
    n_pos = len(positives)
    ax.set_title(f"이 Scenario의 Sample {len(samples)}건 (Positive {n_pos} · Negative {len(samples) - n_pos}) 중 {len(picked)}건",
                 loc="left", fontsize=12.5)
    handles = [Patch(color=REF, label="Past 1 s (Input)"), Patch(color=TGT, alpha=.28, label="Future 3 s (Label 확인)"),
               Line2D([0], [0], marker="*", ls="none", markerfacecolor=TGT, markeredgecolor=INK, markersize=13,
                      label="Entry")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .02), ncol=3, fontsize=10.5)
    save(fig, output_dir, "05_anchor_timeline")


def plot_label_rule(output_dir, scene, items, tracks, row):
    ref, tgt = tracks[int(row["reference_id"])], tracks[int(row["target_id"])]
    t0, entry = int(row["anchor_frame"]), int(row["entry_frame"])
    bounds = maneuver_bounds(tracks, row)
    map_height = 12 * np.ptp(bounds[1]) / np.ptp(bounds[0])
    fig = plt.figure(figsize=(12, map_height + 3.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[map_height, 2.5], hspace=.32, wspace=.16,
                            left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(grid[0, :])
    draw_maneuver(ax, items, scene, tracks, row, bounds)

    frames = np.arange(t0 - HISTORY, t0 + HORIZON + 1)
    rel = relative_track(ref, tgt, frames)
    seconds = (frames - t0) / 10
    lead = (entry - t0) / 10
    panels = ((rel[:, 0], "Target의 앞뒤 거리 (m)", (0, 40), "Rule: Reference 앞 0~40 m"),
              (rel[:, 1], "Target의 좌우 거리 (m)", (-1.75, 1.75), "참고: 같은 차로 폭 ±1.75 m"))
    for i, (values, ylabel, (low, high), text) in enumerate(panels):
        ax = fig.add_subplot(grid[1, i])
        for bound in (low, high):
            ax.axhline(bound, color="#2a9d6f", lw=1, ls=(0, (4, 3)))
        ax.text(2.97, high, text, ha="right", va="top" if i == 0 else "bottom", fontsize=10.5, color="#2a7a58")
        ax.plot(seconds[seconds <= 0], values[seconds <= 0], color=TGT, lw=3)
        ax.plot(seconds[seconds >= 0], values[seconds >= 0], color=TGT, lw=1.8, ls=(0, (3, 2.5)))
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(lead, color=INK, lw=1, ls=(0, (2, 2)))
        ax.set(xlim=(-1, 3), xlabel="예측 시점(t = 0)으로부터 시간 (s)")
        ax.set_title(ylabel, loc="left", fontsize=12.5)
        grid_lines(ax)
        ax.set_ylim(*((-5, 45) if i == 0 else (-2.5, max(4.0, values.max() + .8))))
        ax.text(lead + .05, ax.get_ylim()[0], f" Entry +{lead:.1f} s", fontsize=10.5, va="bottom")
    legend_below(fig, maneuver_legend(), ncol=3, y=-.1)
    save(fig, output_dir, "06_label_rule")


def plot_model_inputs(output_dir, sequence, tabular, names):
    seconds = np.arange(-10, 1) / 10
    fig = plt.figure(figsize=(6.4, 5.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax = fig.add_subplot(grid[0, :])
    for i, (color, name) in enumerate(((REF, "Reference"), (TGT, "Target"))):
        ax.plot(sequence[:, i, 0], sequence[:, i, 1], color=color, lw=2.2, marker="o", ms=5, label=name)
    ax.annotate("t = 0", sequence[-1, 1, :2], xytext=(6, 6), textcoords="offset points", fontsize=10.5, color=TGT)
    ax.annotate("t = -1 s", sequence[0, 1, :2], xytext=(-6, 6), textcoords="offset points", fontsize=10.5, color=TGT,
                ha="right")
    ax.set(xlabel="x: 앞뒤 (m)", ylabel="y: 좌우 (m)", ylim=(-1.5, 5))
    ax.set_title("Sequence: 11 frames × 2 vehicles × 7 values (x, y, vx, vy, yaw, length, width)", loc="left", fontsize=11.5)
    grid_lines(ax, "both")
    ax.legend(loc="lower right", ncol=2, fontsize=10.5)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(seconds, sequence[:, 0, 3], color=REF, lw=2, marker="o", ms=3.5)
    ax.plot(seconds, sequence[:, 1, 3], color=TGT, lw=2, marker="o", ms=3.5)
    ax.set(xlabel="time (s)", title="vy: 좌우 속도 (m/s)")
    ax.title.set_fontsize(12)
    grid_lines(ax)

    ax = fig.add_subplot(grid[1, 1])
    order = [0, 1, 3, 6, 7]
    labels = [names[i] for i in order]
    values = tabular[order]
    bars = ax.barh(labels, values, color=[BLUE_MID if v >= 0 else TGT for v in values], height=.6)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10.5)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_xlim(min(0, values.min()) - 6, values.max() * 1.35)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Tabular: Basic 8개 중 5개", fontsize=12)
    save(fig, output_dir, "07_model_inputs")


def plot_bev_input(output_dir, packed):
    bev = np.unpackbits(packed, axis=-1, count=96)
    titles = ["Lane centerline", "Road line + Road edge", "Reference", "Target", "Other vehicles"]
    colors = ["#8ea2b5", "#ffffff", REF, TGT, OTHER]
    extent = (-24.5, 71.5, -23.5, 24.5)
    fig = plt.figure(figsize=(6.4, 5.4), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[2.2, 1, 1])
    composite = np.tile(np.array(to_rgb(ROAD)), (48, 96, 1))
    for i in (0, 1, 4, 2, 3):  # 지도 → 주변 차량 → 기준·상대 차량 순서로 덮어 그린다.
        composite[bev[i].astype(bool)] = to_rgb(colors[i])
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(composite, extent=extent, interpolation="nearest")
    ax.set_title("BEV: 5 channels 합성 (48 × 96 grid, 1 cell = 1 m)", loc="left", fontsize=12)
    ax.set(xlabel="x: 앞뒤 (m)", ylabel="y: 좌우 (m)")
    ax.tick_params(labelsize=10)
    spots = [grid[1, 0], grid[1, 1], grid[1, 2], grid[2, 0], grid[2, 1]]
    for spot, layer, title, color in zip(spots, bev, titles, colors):
        ax = fig.add_subplot(spot)
        ax.imshow(layer, cmap=ListedColormap([ROAD, color]), extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="normal", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(grid[2, 2])
    ax.axis("off")
    ax.text(0, .5, "channel별 0/1 값\n현재(t = 0)만 포함", fontsize=10.5, color=MUTED, va="center")
    save(fig, output_dir, "08_bev_input")


def plot_dataset_build(output_dir, info):
    splits = ("train", "validation", "test")
    proc, counts = info["processing"], info["counts"]
    total = lambda key: sum(proc[s].get(key, 0) for s in splits)
    fig = plt.figure(figsize=(12, 4.3), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, .9])

    ax = fig.add_subplot(grid[0])
    stages = [("읽은 Scenario", total("scenarios_read")), ("Candidate pair가 있는 Scenario", total("candidate_scenes")),
              ("Label이 붙은 Scenario", total("labeled_scenes"))]
    bars = ax.barh([s for s, _ in stages], [n for _, n in stages], color=[BLUE_LIGHT, BLUE_MID, BLUE_DARK], height=.6)
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, stages[0][1] * 1.35)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Scenario (원본 파일 64개)", loc="left")

    ax = fig.add_subplot(grid[1])
    t = proc["train"]
    labeled = t["candidate_windows"] - t["uncertain_windows"]
    stages = [("Candidate (pair × anchor)", t["candidate_windows"], BLUE_LIGHT),
              ("Uncertain (제외)", t["uncertain_windows"], "#d5dbe1"),
              ("Rule로 Label 확정", labeled, BLUE_MID),
              ("Positive event + Negative pair", t["positive_events"] + t["clean_negative_pairs"], BLUE_DARK),
              ("최종 train (중복 제거 후)", counts["train"]["samples"], INK)]
    bars = ax.barh([s for s, _, _ in stages], [n for _, n, _ in stages], color=[c for _, _, c in stages], height=.6)
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, stages[0][1] * 1.35)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("train Sample (원본 파일 40개)", loc="left")

    ax = fig.add_subplot(grid[2])
    sizes = [counts[s]["samples"] for s in splits]
    ax.bar(splits, sizes, color=[BLUE_DARK, "#c3ccd5", "#c3ccd5"], width=.62)
    pos = counts["train"]["positives"]
    ax.bar(["train"], [pos], color=TGT, width=.62, label=f"train Positive {pos:,}건 ({pos / sizes[0]:.2%})")
    for i, (s, n) in enumerate(zip(splits, sizes)):
        ax.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=11)
    ax.legend(loc="upper right", fontsize=10.5, handlelength=1.2)
    ax.set_xticks(range(3), [f"{s}\n파일 {counts[s]['files']}개" for s in splits])
    ax.set_ylim(0, max(sizes) * 1.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("최종 Split", loc="left")
    save(fig, output_dir, "09_dataset_build")

