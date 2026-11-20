# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scikit-learn==1.9.1", "matplotlib==3.10.9", "protobuf>=6.31,<7"]
# ///
"""Waymo 원본 장면 하나에서 학습 표본이 만들어지는 과정과 전체 데이터셋 구축 과정을 그림 7장으로 저장한다."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 프로젝트 폴더의 src 패키지를 불러오기 위해

from src.dataset import BASIC_FEATURES, read_csv
from src.paths import DATASET_DIR, FIGURES_DIR, RAW_SAMPLES_DIR
from src.plots.pipeline import (
    configure_plots, plot_dataset_build, plot_label, plot_pair_selection, plot_raw_contents,
    plot_raw_scene, plot_saved_bev, plot_saved_sample,
)
from src.raw_scenario import load_scene, map_items, track_states

RAW_DIR = RAW_SAMPLES_DIR
DATA_DIR = DATASET_DIR
OUTPUT_DIR = FIGURES_DIR / "data_pipeline"
SCENE_ID = "b185d20a6504239c"           # 차량·보행자·자전거와 지도 요소가 다양한 예시 장면
TARGET_ID = "1240"                      # 이 장면에서 나온 양성 표본의 상대 차량 ID


def main():
    configure_plots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene, _ = load_scene(RAW_DIR, SCENE_ID)
    items = map_items(scene)
    tracks = {t.id: track_states(t) for t in scene.tracks}
    sdc_id = scene.tracks[scene.sdc_track_index].id

    labels = {r["sample_id"]: r for r in read_csv(DATA_DIR / "labels.csv")}
    metadata = [r | labels[r["sample_id"]] for r in read_csv(DATA_DIR / "samples.csv")]   # 표본 정보 + 최종 라벨
    samples = sorted([r for r in metadata if r["scenario_id"] == SCENE_ID],
                     key=lambda r: (int(r["anchor_frame"]), r["reference_type"] != "waymo_sdc", r["sample_id"]))
    main_row = next((r for r in samples if r["target_id"] == TARGET_ID and r["label"] == "1"),
                    next(r for r in samples if r["label"] == "1"))
    info = json.loads((DATA_DIR / "dataset_info.json").read_text(encoding="utf-8-sig"))
    with np.load(DATA_DIR / "inputs.npz", allow_pickle=False) as arrays:
        index = arrays["sample_id"].tolist().index(main_row["sample_id"])
        sequence, tabular, packed = (arrays[k][index] for k in ("X_sequence", "X_basic", "X_bev_packed"))

    plot_raw_scene(OUTPUT_DIR, scene, items, tracks, sdc_id)
    plot_raw_contents(OUTPUT_DIR, scene, tracks)
    plot_pair_selection(OUTPUT_DIR, scene, items, tracks, samples, main_row)
    plot_label(OUTPUT_DIR, items, tracks, main_row)
    plot_saved_sample(OUTPUT_DIR, tracks, main_row, sequence, tabular, BASIC_FEATURES)
    plot_saved_bev(OUTPUT_DIR, main_row, packed)
    plot_dataset_build(OUTPUT_DIR, info, list(labels.values()))


if __name__ == "__main__":
    main()
