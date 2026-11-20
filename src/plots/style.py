"""모든 그림이 같은 글꼴, 색, 크기, 저장 방식을 쓰도록 모은 설정."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 차량 색은 검토 Colab 화면과 같다.
REF = "#4f8ff0"          # 기준 차량
TGT = "#ff8a3d"          # 상대 차량, 끼어들기(1)
NEG = "#7d8d9e"          # 끼어들기 없음(0)
OTHER = "#8a9bb0"        # 주변 차량
PEDESTRIAN, CYCLIST = "#5fcf8a", "#c79bf2"

# 차트 색
INK = "#1f2933"          # 제목, 숫자
MUTED = "#66727f"        # 축 눈금, 보조 설명
GRID = "#e5e9ed"         # 눈금선
AXIS = "#b9c2cb"         # 축선
BLUE_DARK, BLUE_MID, BLUE_LIGHT = "#1f4e79", "#4f81b0", "#a9c3dc"
HIGHLIGHT = TGT

# 모델 색: 단순한 모델일수록 옅게
MODEL_COLORS = {
    "logistic": "#c3ccd5",
    "logistic_balanced": "#9aa7b4",
    "random_forest": "#6f8fae",
    "boosting": "#3d6b96",
    "boosting_balanced": "#1f4e79",
}
MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "logistic_balanced": "Logistic Regression (class weight)",
    "random_forest": "Random Forest",
    "boosting": "Gradient Boosting",
    "boosting_balanced": "Gradient Boosting (class weight)",
}
INPUT_NAMES = {"basic_8": "Basic 8", "motion_64": "Motion 64", "selected_76": "Selected 76", "engineered_99": "All 99"}

DPI = 200
FULL_WIDTH = 12.0        # 슬라이드 가로 전체에 넣는 그림의 폭(인치)


def configure_plots():
    font = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font.exists():  # macOS 한글 글꼴. 다른 OS에서는 설치된 한글 글꼴 이름(예: NanumGothic)을 쓴다.
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "NanumGothic"
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False, "font.size": 12,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": .9,
        "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlepad": 10, "axes.labelsize": 12,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 11, "lines.solid_capstyle": "round",
    })


def grid_lines(ax, axis="y"):
    """옅은 눈금선을 데이터 뒤에 둔다."""
    ax.grid(axis=axis, color=GRID, lw=.8)
    ax.set_axisbelow(True)


def note(fig, text):
    """그림 아래 왼쪽에 한 줄 설명을 넣는다."""
    fig.text(.01, .005, text, ha="left", va="bottom", fontsize=10.5, color=MUTED)


def save(fig, output_dir, name):
    path = Path(output_dir) / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    print("저장:", path, flush=True)
