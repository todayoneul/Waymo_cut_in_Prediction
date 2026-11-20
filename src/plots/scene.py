"""검토 Colab 화면과 같은 방식으로 도로 장면을 그린다.

어두운 도로 위에 흰색·노란색 차선, 보라색 중앙 분리대, 민트색 도로 경계, 회색 점선 차로 중심선을 그리고
기준 차량은 파란색, 상대 차량은 주황색, 주변 차량은 회색으로 표시한다.
"""
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from .style import CYCLIST, OTHER, PEDESTRIAN, REF, TGT

ROAD = "#253546"         # 도로 배경
LABEL_SHADOW = [path_effects.withStroke(linewidth=3, foreground=ROAD)]

# 선 종류: (색, 선 모양, 굵기)
LINE_STYLE = {
    "center": ("#8ea2b5", (0, (1, 2.6)), .9),
    "edge": ("#67c4aa", "-", 1.6),
    "median": ("#dda3f3", "-", 2.2),
    "white_solid": ("#ffffff", "-", 1.4),
    "white_dash": ("#ffffff", (0, (5, 4)), 1.4),
    "yellow_solid": ("#ffcf64", "-", 1.4),
    "yellow_dash": ("#ffcf64", (0, (5, 4)), 1.4),
    "unknown": ("#9cb1c1", "-", 1.0),
}


def line_kind(kind, line_type):
    """Waymo 지도 요소 종류와 선 번호를 그림의 선 종류로 바꾼다 (검토 Colab과 같은 규칙)."""
    if kind == "lane":
        return "center"
    if kind == "road_edge":
        return "median" if line_type == 2 else "edge"
    if line_type in (1, 2, 3):
        return "white_dash" if line_type == 1 else "white_solid"
    if line_type in (4, 5, 6, 7, 8):
        return "yellow_dash" if line_type in (4, 5) else "yellow_solid"
    return "unknown"


def road_axes(ax, xlim, ylim, show_ticks=False, scale_bar=True):
    """도로 배경을 칠하고 축 범위를 맞춘다. 눈금을 숨기면 10 m 축척 막대를 넣는다."""
    ax.set_facecolor(ROAD)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    for spine in ax.spines.values():
        spine.set_visible(show_ticks)
        spine.set_color(ROAD)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if scale_bar:
        x1 = xlim[1] - .02 * (xlim[1] - xlim[0])
        y0 = ylim[0] + .07 * (ylim[1] - ylim[0])
        ax.plot([x1 - 10, x1], [y0, y0], color="#dae5ee", lw=2, zorder=20, solid_capstyle="butt")
        ax.annotate("10 m", (x1 - 5, y0), xytext=(0, 3), textcoords="offset points", color="#dae5ee",
                    fontsize=10, ha="center", va="bottom", zorder=20)


def draw_road(ax, items, transform=lambda xy: xy, show_centers=True):
    """지도 요소(map_items 결과)를 그린다. 횡단보도는 도로보다 조금 밝게 칠한다."""
    for kind, line_type, points in items:
        xy = transform(points)
        if kind in ("lane", "road_line", "road_edge") and len(xy) >= 2:
            style = line_kind(kind, line_type)
            if style == "center" and not show_centers:
                continue
            color, dash, width = LINE_STYLE[style]
            ax.plot(*xy.T, color=color, ls=dash, lw=width, zorder=1 if style == "center" else 2)
        elif kind == "crosswalk" and len(xy) >= 3:
            ax.add_patch(Polygon(xy, closed=True, facecolor="#33485c", edgecolor="none", zorder=0))


def outline(state, transform):
    """차량 상자(앞쪽이 뾰족한 오각형)의 꼭짓점. state = x, y, 방향, vx, vy, 길이, 너비, 유효."""
    x, y, heading, _, _, length, width, _ = state
    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-forward[1], forward[0]])
    center = np.array([x, y])
    half_l, half_w = max(length, 1) / 2, max(width, .5) / 2
    corners = [center + half_l * forward + half_w * left, center + (half_l + .5) * forward,
               center + half_l * forward - half_w * left, center - half_l * forward - half_w * left,
               center - half_l * forward + half_w * left]
    return transform(np.array(corners))


def draw_vehicle(ax, state, transform, color, label=None, z=6, label_side=1):
    """차량을 그린다. 이름표는 label_side 쪽(+1 위, -1 아래)에 화면 크기와 상관없이 일정한 간격으로 붙인다."""
    ax.add_patch(Polygon(outline(state, transform), closed=True, facecolor=color,
                         edgecolor="white" if color in (REF, TGT) else "#c9d4de",
                         lw=1.2 if color in (REF, TGT) else .5, zorder=z))
    if label:
        x, y = transform(np.array([state[:2]]))[0]
        ax.annotate(label, (x, y), xytext=(0, 13 * label_side), textcoords="offset points", color=color,
                    fontsize=11.5, fontweight="bold", ha="center", va="bottom" if label_side > 0 else "top",
                    zorder=z + 2, path_effects=LABEL_SHADOW)


def draw_agents(ax, scene, tracks, frame, transform, roles=None, show_people=True):
    """한 시점의 모든 차량(주변 차량은 회색)과 보행자·자전거를 그린다. roles = {차량 ID: (색, 이름표)}.

    역할 차량이 두 대이면 서로의 이름표가 겹치지 않도록 상대 차량 쪽의 반대편에 이름표를 붙인다.
    """
    roles = roles or {}
    sides = {}
    if len(roles) == 2:
        first, second = list(roles)
        y_first, y_second = (transform(np.array([tracks[v][frame, :2]]))[0][1] for v in (first, second))
        sides = {first: -1 if y_second > y_first else 1, second: 1 if y_second > y_first else -1}
    for track in scene.tracks:
        state = tracks[track.id][frame]
        if state[7] <= 0:
            continue
        if track.object_type == 1:
            color, label = roles.get(track.id, (OTHER, None))
            draw_vehicle(ax, state, transform, color, label, z=8 if track.id in roles else 6,
                         label_side=sides.get(track.id, 1))
        elif show_people and track.object_type in (2, 3):
            xy = transform(np.array([state[:2]]))[0]
            ax.scatter(*xy, s=22, color=PEDESTRIAN if track.object_type == 2 else CYCLIST,
                       edgecolor="white", lw=.5, zorder=7)


def road_legend(include_people=False):
    handles = [Patch(facecolor=REF, edgecolor="white", label="Reference vehicle"),
               Patch(facecolor=TGT, edgecolor="white", label="Target vehicle"),
               Patch(facecolor=OTHER, edgecolor="#c9d4de", label="Other vehicle")]
    if include_people:
        handles += [Line2D([0], [0], marker="o", ls="none", markerfacecolor=PEDESTRIAN, markeredgecolor="white",
                           markersize=8, label="Pedestrian"),
                    Line2D([0], [0], marker="o", ls="none", markerfacecolor=CYCLIST, markeredgecolor="white",
                           markersize=8, label="Cyclist")]
    line = lambda key, label: Line2D([0], [0], color=LINE_STYLE[key][0], ls=LINE_STYLE[key][1],
                                     lw=LINE_STYLE[key][2] + .6, label=label)
    return handles + [line("white_dash", "Road line (흰색·노란색)"), line("median", "Median (중앙 분리대)"),
                      line("edge", "Road edge"), line("center", "Lane centerline")]


def view_bounds(points, margin_x=(8, 10), margin_y=4.5, min_half_height=7):
    """두 차량 경로가 모두 보이는 범위. points = (N, 2) 좌표."""
    low, high = points.min(axis=0), points.max(axis=0)
    middle, half = (low[1] + high[1]) / 2, max((high[1] - low[1]) / 2 + margin_y, min_half_height)
    return (low[0] - margin_x[0], high[0] + margin_x[1]), (middle - half, middle + half)


def legend_below(owner, handles, ncol=None, y=-.03):
    """도로 범례를 그림(owner = 축 또는 figure) 바로 아래에 한 줄로 놓는다. 흰 차선 기호가 보이도록 범례 상자는 도로색이다."""
    legend = owner.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y), ncol=ncol or len(handles),
                          frameon=True, framealpha=1, facecolor=ROAD, edgecolor=ROAD, labelcolor="white",
                          fontsize=11, handlelength=2.2, columnspacing=1.6, borderpad=.7)
    legend.get_frame().set_boxstyle("round,pad=0.2,rounding_size=0.3")
    return legend
