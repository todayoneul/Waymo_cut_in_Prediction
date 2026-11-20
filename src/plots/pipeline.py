"""원본 장면 → 표본 선택 → 라벨 → 저장된 입력 → 전체 데이터셋으로 이어지는 구축 과정 그림."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from ..raw_scenario import frame_of, vehicle_outline

HISTORY, HORIZON = 10, 30               # 10 Hz 기준 과거 1초(입력), 미래 3초(정답 확인)

REF, TGT = "#1f6fb2", "#e0702e"
VEHICLE, PEDESTRIAN, CYCLIST = "#7f8c99", "#2e8b57", "#8e44ad"
INK, MUTED, GRID = "#1d2b36", "#5b6b77", "#e3e8ec"
TYPE_NAMES = {1: "차량", 2: "보행자", 3: "자전거", 4: "기타"}
TYPE_COLORS = {1: VEHICLE, 2: PEDESTRIAN, 3: CYCLIST, 4: "#b0b0b0"}
MAP_NAMES = {"lane": "차로 중심선", "road_line": "도로 표시선", "road_edge": "도로 경계",
             "crosswalk": "횡단보도", "speed_bump": "과속방지턱", "driveway": "진입로",
             "stop_sign": "정지 표지"}


def configure_plots():
    font = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font.exists():  # macOS 한글 글꼴. 다른 OS에서는 설치된 한글 글꼴 이름으로 바꾼다.
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "NanumGothic"
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#b8c2ca",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11.5,
        "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
    })


def draw_map(ax, items, transform=lambda p: p):
    for kind, line_type, points in items:
        xy = transform(points)
        if kind == "lane":
            ax.plot(*xy.T, color="#c3ccd4", lw=.7, ls=(0, (2, 2)), zorder=1)
        elif kind == "road_line":
            yellow, broken = line_type in (4, 5, 6, 7, 8), line_type in (1, 4, 5)
            ax.plot(*xy.T, color="#d4a017" if yellow else "#87939c", lw=1.0,
                    ls=(0, (5, 4)) if broken else "-", zorder=2)
        elif kind == "road_edge":
            ax.plot(*xy.T, color="#34424d", lw=1.3, zorder=2)
        elif kind == "stop_sign":
            ax.scatter(*xy.T, marker="8", s=28, color="#c0392b", zorder=3)
        elif len(xy) >= 3:
            fill = {"crosswalk": "#dfe6ec", "speed_bump": "#f5d7b5", "driveway": "#f2f4f6"}[kind]
            ax.add_patch(Polygon(xy, closed=True, facecolor=fill, edgecolor="none", zorder=0))


def map_legend():
    return [Line2D([0], [0], color="#c3ccd4", lw=1, ls=(0, (2, 2)), label="차로 중심선"),
            Line2D([0], [0], color="#87939c", lw=1.2, label="흰색 도로 표시선"),
            Line2D([0], [0], color="#d4a017", lw=1.2, label="노란색 도로 표시선"),
            Line2D([0], [0], color="#34424d", lw=1.5, label="도로 경계"),
            Patch(facecolor="#dfe6ec", label="횡단보도")]


def footnote(fig, text):
    """그림 아래 설명을 축 이름과 겹치지 않게 넣는다."""
    fig.get_layout_engine().set(rect=(0, .04, 1, .96))
    fig.text(.995, .01, text, ha="right", fontsize=8.5, color=MUTED)


def save(fig, output_dir, name):
    fig.savefig(Path(output_dir) / f"{name}.png", dpi=170)
    plt.close(fig)
    print("저장:", Path(output_dir) / f"{name}.png", flush=True)


def plot_raw_scene(output_dir, scene, items, tracks, sdc_id):
    valid_xy = np.vstack([s[s[:, 7] > 0, :2] for s in tracks.values() if (s[:, 7] > 0).any()])
    low, high = valid_xy.min(axis=0) - 12, valid_xy.max(axis=0) + 12
    fig, ax = plt.subplots(figsize=(13, 8.2), layout="constrained")
    fig.suptitle(f"① Waymo 원본 장면 1개 (scenario {scene.scenario_id})\n"
                 f"9초 · 10 Hz · 91개 시점 · 객체 {len(scene.tracks)}개 · 지도 요소 {len(scene.map_features)}개",
                 fontsize=15, fontweight="bold", linespacing=1.5)
    draw_map(ax, items)
    current = scene.current_time_index
    for track in scene.tracks:
        states = tracks[track.id]
        ok = states[:, 7] > 0
        is_sdc = track.id == sdc_id
        color = "#c0392b" if is_sdc else TYPE_COLORS.get(track.object_type, "#b0b0b0")
        ax.plot(states[ok, 0], states[ok, 1], color=color, lw=2.6 if is_sdc else 1.0,
                alpha=1 if is_sdc else .55, zorder=5 if is_sdc else 4)
        if states[current, 7] > 0:
            if track.object_type == 1:
                ax.add_patch(Polygon(vehicle_outline(states[current], lambda p: np.asarray(p), 0),
                                     closed=True, facecolor=color, edgecolor="white", lw=.6,
                                     alpha=1 if is_sdc else .85, zorder=6))
            else:
                ax.scatter(*states[current, :2], s=18, color=color, edgecolor="white", lw=.5, zorder=6)
    sdc = tracks[sdc_id][current]
    ax.annotate("Waymo 차량", sdc[:2], xytext=(14, 14), textcoords="offset points", color="#c0392b",
                fontweight="bold", arrowprops=dict(arrowstyle="-", color="#c0392b", lw=.8))
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal",
           xlabel="원본 지도 좌표 x (m)", ylabel="원본 지도 좌표 y (m)")
    agents = [Line2D([0], [0], color="#c0392b", lw=2.6, label="Waymo 차량(SDC) 궤적"),
              Line2D([0], [0], color=VEHICLE, lw=1.5, label="다른 차량 궤적"),
              Line2D([0], [0], color=PEDESTRIAN, lw=1.5, label="보행자 궤적"),
              Line2D([0], [0], color=CYCLIST, lw=1.5, label="자전거 궤적")]
    ax.legend(handles=agents + map_legend(), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              title="선: 0~9초 전체 이동 · 도형: 공식 기준 시점(1.0초) 위치", title_fontsize=9)
    save(fig, output_dir, "01_raw_scene")


def plot_raw_contents(output_dir, scene, tracks):
    fig = plt.figure(figsize=(13.5, 8.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
    fig.suptitle("② 원본 장면에 들어 있는 정보", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(grid[0, 0])
    kinds = [1, 2, 3, 4]
    counts = [sum(t.object_type == k for t in scene.tracks) for k in kinds]
    bars = ax.barh([TYPE_NAMES[k] for k in kinds], counts, color=[TYPE_COLORS[k] for k in kinds], height=.55)
    ax.bar_label(bars, padding=3)
    ax.invert_yaxis()
    ax.set(title="객체 종류별 개수", xlabel="객체 수")

    ax = fig.add_subplot(grid[0, 1])
    names = list(MAP_NAMES)
    counts = [sum(f.WhichOneof("feature_data") == n for f in scene.map_features) for n in names]
    bars = ax.barh([MAP_NAMES[n] for n in names], counts, color="#5d7a92", height=.55)
    ax.bar_label(bars, padding=3)
    ax.invert_yaxis()
    signals = sum(len(d.lane_states) for d in scene.dynamic_map_states)
    ax.set(title=f"정적 지도 요소 · 신호등 상태 기록 {signals}건", xlabel="요소 수")

    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    ax.set_title("객체 한 개의 한 시점에 저장된 값", loc="left")
    rows = [("center_x, y, z", "박스 중심 위치 (m)"), ("length, width, height", "박스 크기 (m)"),
            ("heading", "진행 방향 (rad)"), ("velocity_x, y", "속도 (m/s)"),
            ("valid", "관측 여부"), ("", ""),
            ("장면 공통", "시간표 91개, Waymo 차량 번호,"), ("", "공식 기준 시점 번호 등")]
    for i, (field, meaning) in enumerate(rows):
        ax.text(0, .9 - i * .115, field, fontsize=10, fontweight="bold", family="monospace" if field and field != "장면 공통" else None)
        ax.text(.52, .9 - i * .115, meaning, fontsize=10, color=MUTED)

    ax = fig.add_subplot(grid[1, :])
    ordered = sorted(scene.tracks, key=lambda t: (t.object_type, int(np.argmax(tracks[t.id][:, 7] > 0))))
    matrix = np.array([np.where(tracks[t.id][:, 7] > 0, t.object_type, 0) for t in ordered])
    ax.imshow(matrix, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#f1f3f5", VEHICLE, PEDESTRIAN, CYCLIST, "#b0b0b0"]), vmin=0, vmax=4)
    current = scene.current_time_index
    ax.axvline(current, color="#c0392b", lw=1.6)
    ax.text(current + .8, -3, "공식 기준 시점 (1.0초)", color="#c0392b", fontsize=9, va="bottom")
    ax.set_xticks(np.arange(0, 91, 10), [f"{t:.0f}" for t in np.arange(0, 9.1, 1)])
    ax.set(title=f"객체 {len(ordered)}개 × 시점 91개의 관측 여부 (색칠 = 관측됨)",
           xlabel="장면 시간 (초)", ylabel="객체 (종류별 정렬)")
    ax.legend(handles=[Patch(color=TYPE_COLORS[k], label=TYPE_NAMES[k]) for k in (1, 2, 3)] +
              [Patch(color="#f1f3f5", label="관측 없음")], loc="upper right", ncol=4, fontsize=9,
              bbox_to_anchor=(1, 1.13))
    save(fig, output_dir, "02_raw_contents")


def plot_pair_selection(output_dir, scene, items, tracks, samples, main):
    ref_states, t0 = tracks[int(main["reference_id"])], int(main["anchor_frame"])
    transform, heading0 = frame_of(ref_states[t0])
    fig = plt.figure(figsize=(14.5, 6.6), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    fig.suptitle("③ 원본에서 표본 고르기: 기준 차량 · 상대 차량 · 기준 시점", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(grid[0])
    draw_map(ax, items, transform)
    for side in (1, -1):  # 상대 차량 후보 영역: 앞뒤 -15~55 m, 옆 2.3~11 m
        ax.add_patch(Rectangle((-15, 2.3 if side > 0 else -11), 70, 8.7, facecolor="#dceaf6",
                               edgecolor="#8fb6d8", lw=1, ls="--", alpha=.55, zorder=0))
    ax.text(-14, 11.6, "상대 차량 후보 영역 (앞뒤 -15~55 m, 옆 2.3~11 m)", color="#3d76a8", fontsize=9)
    target_id = int(main["target_id"])
    for track in scene.tracks:
        state = tracks[track.id][t0]
        if state[7] <= 0:
            continue
        if track.object_type == 1:
            role = REF if track.id == int(main["reference_id"]) else TGT if track.id == target_id else None
            ax.add_patch(Polygon(vehicle_outline(state, transform, heading0), closed=True,
                                 facecolor=role or "#c7cfd6", edgecolor="white", lw=.7, zorder=5))
        else:
            ax.scatter(*transform([state[:2]]).T, s=16, color=TYPE_COLORS[track.object_type], zorder=5)
    for name, vid, color in (("기준 차량 (Waymo)", int(main["reference_id"]), REF), ("상대 차량", target_id, TGT)):
        xy = transform([tracks[vid][t0, :2]])[0]
        ax.annotate(name, xy, xytext=(0, 13), textcoords="offset points", ha="center", color=color,
                    fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    ax.set(xlim=(-28, 68), ylim=(-20, 20), aspect="equal", xlabel="기준 차량 진행 방향 (m)",
           ylabel="기준 차량 왼쪽(+)/오른쪽(-) (m)",
           title=f"기준 시점 {t0 / 10:.1f}초: 기준 차량을 원점에 두고 회전한 좌표")
    rules = ("후보 조건\n· 기준 차량: 초속 2 m 이상, 과거 1초 관측\n· 상대 차량: 같은 방향, 지도상 인접 차로\n"
             "· 보행자·자전거는 상대 차량에서 제외")
    ax.text(.01, .02, rules, transform=ax.transAxes, fontsize=8.8, va="bottom", color=INK,
            bbox=dict(facecolor="white", edgecolor=GRID, pad=5), zorder=20)

    ax = fig.add_subplot(grid[1])
    for t in range(10, 51, 10):
        ax.axvline(t / 10, color="#c9d1d8", lw=1, ls=":", zorder=0)
    ax.axvspan(5.8, 9.0, color="#f3f4f6", zorder=0)
    ax.text(7.4, len(samples) - .35, "미래 3초를\n확인할 수 없어\n기준 시점 불가", ha="center", va="top",
            fontsize=8.5, color=MUTED)
    for row, sample in enumerate(samples):
        t = int(sample["anchor_frame"]) / 10
        y = len(samples) - 1 - row
        ax.barh(y, 1.0, left=t - 1.0, height=.42, color=REF, alpha=.85)
        ax.barh(y, 3.0, left=t, height=.42, color=TGT, alpha=.32)
        if sample["entry_frame"]:
            entry = int(sample["entry_frame"]) / 10
            ax.scatter(entry, y, marker="*", s=150, color=TGT, edgecolor=INK, lw=.6, zorder=4)
        verdict = "끼어들기 (1)" if sample["label"] == "1" else "끼어들기 없음 (0)"
        who = "Waymo" if sample["reference_type"] == "waymo_sdc" else "일반 차량"
        ax.text(9.05, y, f"{verdict}\n{sample['label_source']} 라벨", va="center", fontsize=8.5)
        ax.text(-.05, y, f"기준 #{sample['reference_id']} ({who})\n→ 상대 #{sample['target_id']}",
                va="center", ha="right", fontsize=8.5)
    ax.set_yticks([])
    ax.set(xlim=(0, 9), ylim=(-.7, len(samples) - .3), xlabel="장면 시간 (초)",
           title=f"이 장면에서 실제로 만든 표본 {len(samples)}건")
    ax.legend(handles=[Patch(color=REF, alpha=.85, label="과거 1초 → 모델 입력"),
                       Patch(color=TGT, alpha=.32, label="미래 3초 → 정답 확인"),
                       Line2D([0], [0], marker="*", color="none", markerfacecolor=TGT, markeredgecolor=INK,
                              markersize=12, label="규칙이 확인한 진입 시점"),
                       Line2D([0], [0], color="#c9d1d8", ls=":", label="기준 시점 후보 (1초 간격)")],
              loc="upper center", bbox_to_anchor=(.45, -.12), ncol=2, fontsize=8.8)
    save(fig, output_dir, "03_pair_and_anchor_selection")


def relative_track(reference, target, frames):
    """각 시점마다 기준 차량 좌표계에서 본 상대 차량의 앞뒤/좌우 위치."""
    values = []
    for j in frames:
        transform, _ = frame_of(reference[j])
        values.append(transform([target[j, :2]])[0])
    return np.array(values)


def plot_label(output_dir, items, tracks, main):
    ref, tgt = tracks[int(main["reference_id"])], tracks[int(main["target_id"])]
    t0, entry = int(main["anchor_frame"]), int(main["entry_frame"])
    transform, heading0 = frame_of(ref[t0])
    past, future = np.arange(t0 - HISTORY, t0 + 1), np.arange(t0, t0 + HORIZON + 1)
    fig = plt.figure(figsize=(14.5, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    fig.suptitle("④ 미래 3초를 확인해 정답(라벨) 만들기 — 미래 정보는 모델 입력에 넣지 않음",
                 fontsize=15, fontweight="bold")

    ax = fig.add_subplot(grid[:, 0])
    draw_map(ax, items, transform)
    for states, color, name in ((ref, REF, "기준 차량"), (tgt, TGT, "상대 차량")):
        p, f = transform(states[past, :2]), transform(states[future, :2])
        ax.plot(*p.T, color=color, lw=2.6, zorder=6)
        ax.plot(*f.T, color=color, lw=1.6, ls="--", alpha=.75, zorder=6)
        ax.scatter(*f[::5].T, s=12, color=color, zorder=7)
        ax.add_patch(Polygon(vehicle_outline(states[t0], transform, heading0), closed=True,
                             facecolor=color, edgecolor="white", lw=.8, zorder=8))
        ax.add_patch(Polygon(vehicle_outline(states[entry], transform, heading0), closed=True,
                             facecolor="none", edgecolor=color, lw=1.4, ls="--", zorder=8))
    ax.scatter(*transform([tgt[entry, :2]]).T, marker="*", s=230, color=TGT, edgecolor=INK, lw=.7, zorder=9)
    ax.set(xlim=(-25, 60), ylim=(-16, 16), aspect="equal", xlabel="기준 시점의 기준 차량 진행 방향 (m)",
           ylabel="좌우 (m)", title="실선 = 과거 1초 (입력) · 점선 = 미래 3초 (정답 확인용)")
    ax.legend(handles=[Patch(color=REF, label="기준 차량"), Patch(color=TGT, label="상대 차량"),
                       Patch(facecolor="none", edgecolor=INK, ls="--", label="진입 확인 시점의 위치"),
                       Line2D([0], [0], marker="*", color="none", markerfacecolor=TGT,
                              markeredgecolor=INK, markersize=13, label="진입 확인 지점")],
              loc="lower left", fontsize=8.8, ncol=2, frameon=True, facecolor="white", edgecolor=GRID)

    frames = np.arange(t0 - HISTORY, t0 + HORIZON + 1)
    rel = relative_track(ref, tgt, frames)
    seconds = (frames - t0) / 10
    lead = (entry - t0) / 10
    for row, (values, ylabel, band, band_label) in enumerate((
            (rel[:, 0], "앞뒤 거리 (m)", (0, 40), "규칙: 기준 차량 앞 0~40 m"),
            (rel[:, 1], "좌우 위치 (m)", (-1.75, 1.75), "같은 차로 범위 근사 (±1.75 m)"))):
        ax = fig.add_subplot(grid[row, 1])
        ax.axvspan(-1, 0, color=REF, alpha=.08)
        ax.axvspan(0, 3, color=TGT, alpha=.06)
        ax.axhspan(*band, color="#2a9d6f", alpha=.07)
        ax.text(2.97, band[1], band_label, ha="right", va="bottom", fontsize=8.5, color="#2a7a58")
        ax.plot(seconds, values, color=TGT, lw=2)
        ax.axvline(lead, color=INK, lw=1, ls="--")
        ax.axvline(0, color=MUTED, lw=.8)
        ax.set(ylabel=ylabel, xlim=(-1, 3))
        ax.grid(color=GRID)
        if row == 0:
            ax.set_title(f"상대 차량의 위치 변화 · 진입 확인 +{lead:.1f}초 → 라벨 1", loc="left")
            ax.text(-.5, ax.get_ylim()[1], "입력 구간", ha="center", va="top", color=REF, fontsize=9)
            ax.text(1.5, ax.get_ylim()[1], "정답 확인 구간", ha="center", va="top", color=TGT, fontsize=9)
        else:
            ax.set_xlabel("기준 시점으로부터 시간 (초)")
    footnote(fig, "실제 규칙은 HD 지도의 차로 번호로 판정합니다: 같은 차로 진입 + 기준 차량 앞 0~40 m + 진행 방향 유사 + 3프레임 유지. "
             "오른쪽 그래프의 녹색 범위는 이해를 돕기 위한 근사입니다.")
    save(fig, output_dir, "04_label_from_future")


def plot_saved_sample(output_dir, tracks, main, sequence, tabular, names):
    ref, tgt = tracks[int(main["reference_id"])], tracks[int(main["target_id"])]
    t0 = int(main["anchor_frame"])
    transform, heading0 = frame_of(ref[t0])
    frames = np.arange(t0 - HISTORY, t0 + 1)
    recomputed = np.stack([transform(ref[frames, :2]), transform(tgt[frames, :2])], axis=1)
    error = float(np.abs(recomputed - sequence[:, :, :2]).max())
    seconds = (frames - t0) / 10

    fig = plt.figure(figsize=(14.5, 8.6), layout="constrained")
    fig.suptitle(f"⑤ 데이터셋에 저장된 표본 1건 · {main['sample_id']} · label = {main['label']} "
                 f"({'끼어들기' if main['label'] == '1' else '없음'}) · {main['split']}",
                 fontsize=15, fontweight="bold")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
    left, right = top.subplots(1, 2, width_ratios=[1.45, 1])

    ax = left
    for i, (color, name) in enumerate(((REF, "기준 차량"), (TGT, "상대 차량"))):
        ax.plot(sequence[:, i, 0], sequence[:, i, 1], color=color, lw=2, marker="o", ms=5, label=f"{name} (저장값)")
    ax.scatter(recomputed[:, :, 0].ravel(), recomputed[:, :, 1].ravel(), s=70, facecolor="none",
               edgecolor=INK, lw=.8, label="원본에서 다시 계산한 위치")
    ax.set(ylim=(-1.5, 5), xlabel="앞뒤 (m)", ylabel="좌우 (m)",
           title=f"X_sequence 위치: 11시점 × 차량 2대  ·  원본과의 최대 차이 {error:.4f} m")
    ax.grid(color=GRID)
    ax.legend(fontsize=9, loc="upper left")

    ax = right
    labels = {"forward_m": "앞뒤 거리 (m)", "lateral_m": "좌우 거리 (m)",
              "relative_forward_mps": "앞뒤 상대속도 (m/s)", "relative_lateral_mps": "좌우 상대속도 (m/s)",
              "target_forward_1s_m": "상대 1초 앞뒤 이동 (m)", "target_lateral_1s_m": "상대 1초 좌우 이동 (m)",
              "reference_speed_mps": "기준 차량 속력 (m/s)", "target_speed_mps": "상대 차량 속력 (m/s)"}
    bars = ax.barh([labels[n] for n in names], tabular, color=["#5d7a92" if v >= 0 else "#c27b5a" for v in tabular],
                   height=.55)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.invert_yaxis()
    ax.set(title="X_basic: 기본 숫자 8개", xlim=(min(0, tabular.min()) - 4, tabular.max() * 1.25 + 1))

    axes = bottom.subplots(1, 5)
    bottom.suptitle("X_sequence 시간 변화 (파란색 = 기준 차량, 주황색 = 상대 차량)", fontsize=11.5, fontweight="bold")
    fields = [(0, "x (앞뒤, m)"), (1, "y (좌우, m)"), (2, "vx (m/s)"), (3, "vy (m/s)"), (4, "방향 (rad)")]
    for ax, (index, title) in zip(axes, fields):
        ax.plot(seconds, sequence[:, 0, index], color=REF, marker="o", ms=3, lw=1.6)
        ax.plot(seconds, sequence[:, 1, index], color=TGT, marker="o", ms=3, lw=1.6)
        ax.set(title=title, xlabel="시간 (초)")
        ax.grid(color=GRID)
    footnote(fig, "X_sequence의 나머지 두 값은 차량 길이·너비입니다. 모든 좌표는 기준 시점의 기준 차량을 원점으로 회전한 값입니다.")
    save(fig, output_dir, "05_saved_sample_sequence_tabular")


def plot_saved_bev(output_dir, main, packed):
    bev = np.unpackbits(packed, axis=-1, count=96)
    titles = ["차로 중심선", "도로 표시선 + 도로 경계", "기준 차량", "상대 차량", "주변 차량 (차량만)"]
    colors = ["#8aa1b4", "#34424d", REF, TGT, "#7f8c99"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), layout="constrained")
    fig.suptitle("⑥ X_bev: 기준 시점의 지도와 차량을 위에서 본 5채널 격자 (1칸 = 1 m, 48 × 96)",
                 fontsize=15, fontweight="bold")
    extent = (-24.5, 71.5, -23.5, 24.5)
    for ax, layer, title, color in zip(axes.flat, bev, titles, colors):
        ax.imshow(layer, cmap=ListedColormap(["white", color]), extent=extent, interpolation="nearest")
        ax.set(title=title, xlabel="앞뒤 (m)", ylabel="좌우 (m)")
    ax = axes.flat[5]
    rgb = np.ones((48, 96, 3))
    for i in (0, 1, 4, 2, 3):  # 지도 → 주변 차량 → 기준·상대 차량 순서로 덮어 그린다.
        rgb[bev[i].astype(bool)] = matplotlib.colors.to_rgb(colors[i])
    ax.imshow(rgb, extent=extent, interpolation="nearest")
    ax.set(title="5개 채널 합성", xlabel="앞뒤 (m)", ylabel="좌우 (m)")
    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    footnote(fig, "현재(0초) 정보만 담습니다. 보행자·자전거와 신호등 상태는 들어 있지 않습니다.")
    save(fig, output_dir, "06_saved_sample_bev")


def plot_dataset_build(output_dir, info, labels):
    splits = ("train", "validation", "test")
    proc, counts = info["processing"], info["counts"]
    files = {s: counts[s]["files"] for s in splits}
    fig = plt.figure(figsize=(14.5, 7.6), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 1])
    fig.suptitle(f"⑦ 같은 과정을 Waymo 원본 파일 {sum(files.values())}개에 반복해 데이터셋 구축", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(grid[0])
    total = lambda key: sum(proc[s].get(key, 0) for s in splits)
    stages = [("읽은 원본 장면", total("scenarios_read")), ("후보 차량 쌍이 있는 장면", total("candidate_scenes")),
              ("양성 또는 명확한 음성이 있는 장면", total("labeled_scenes"))]
    bars = ax.barh([s for s, _ in stages], [n for _, n in stages], color=["#9fb4c6", "#6f90ab", "#3f6f95"], height=.55)
    ax.bar_label(bars, fmt="{:,.0f}", padding=4)
    ax.invert_yaxis()
    ax.set(title="장면 단위 (전체 분할)", xlabel="장면 수", xlim=(0, stages[0][1] * 1.3))

    ax = fig.add_subplot(grid[1])
    t = proc["train"]
    stages = [("차량 쌍 × 기준 시점 후보", t["candidate_windows"], "#9fb4c6"),
              ("그중 판단 보류 (관측 부족·차로 애매 등)", t["uncertain_windows"], "#c9d1d8"),
              ("양성: 같은 끼어들기를 1건으로 병합", t["positive_events"], TGT),
              ("명확한 음성: 차량 쌍마다 1건", t["clean_negative_pairs"], "#5d7a92"),
              ("최종 train 표본 (중복·사람 판정 반영)", counts["train"]["samples"], INK)]
    bars = ax.barh([s for s, _, _ in stages], [n for _, n, _ in stages], color=[k for _, _, k in stages], height=.55)
    ax.bar_label(bars, fmt="{:,.0f}", padding=4)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set(title=f"train 표본 단위 (로그 축) · 양성 {counts['train']['positives']:,}건", xlabel="개수",
           xlim=(max(1, min(n for _, n, _ in stages) / 3), stages[0][1] * 4))

    ax = fig.add_subplot(grid[2])
    sizes = [counts[s]["samples"] for s in splits]
    pos = counts["train"]["positives"]
    ax.bar(splits, sizes, color=["#5d7a92", "#c9d1d8", "#c9d1d8"])
    ax.bar(["train"], [pos], color=TGT, label="train 양성 (1)")
    excluded = info.get("excluded", {})
    removed = {"validation": excluded.get("same_drive_as_train", 0) + excluded.get("input_matches_train", 0),
               "test": excluded.get("same_drive_as_earlier_split", 0) + excluded.get("input_matches_earlier_split", 0)}
    for i, (s, n) in enumerate(zip(splits, sizes)):
        note = f"양성 {pos / n:.1%}" if s == "train" else f"라벨 비공개\n같은 주행 제외 {removed[s]:,}"
        ax.text(i, n, f"{n:,}\n{note}", ha="center", va="bottom", fontsize=8.5)
    ax.set(title="최종 분할 (원본 파일 단위)", ylabel="표본 수", ylim=(0, max(sizes) * 1.3),
           xticks=range(3), xticklabels=[f"{s}\n파일 {files[s]}개" for s in splits])
    ax.legend(fontsize=9, loc="upper left")
    footnote(fig, "분할은 데이터를 읽기 전에 원본 파일 번호로 정했습니다. 명확한 음성은 줄이지 않고 모두 저장했고, "
                  "앞 분할과 같은 주행 구간은 뒤 분할에서 뺐고, validation·test는 표본 수만 표시합니다.")
    save(fig, output_dir, "07_dataset_build")
