# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scikit-learn==1.9.1", "matplotlib==3.10.9", "protobuf>=6.31,<7"]
# ///
"""Waymo 원본 장면에서 학습 표본이 만들어지는 과정과 전체 데이터셋 구축 결과를 그림 9장으로 저장한다."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 프로젝트 폴더의 src 패키지를 불러오기 위해

from src.dataset import BASIC_FEATURES, read_csv
from src.paths import DATASET_DIR, FIGURES_DIR, RAW_SAMPLES_DIR
from src.plots.pipeline import (
    plot_anchor_timeline, plot_bev_input, plot_candidate_pairs, plot_cutin_example, plot_dataset_build,
    plot_label_rule, plot_model_inputs, plot_raw_contents, plot_raw_scene,
)
from src.plots.style import configure_plots
from src.raw_scenario import load_scene, map_items, track_states

OUTPUT_DIR = FIGURES_DIR / "data_pipeline"
MAIN_SCENE, MAIN_TARGET = "b185d20a6504239c", "1240"        # 원본 구조·표본 선택·라벨 규칙·입력 그림에 쓰는 장면
EXAMPLE_SCENE, EXAMPLE_TARGET = "dc406d5eaf678338", "3024"  # 문제 정의 그림에 쓰는 다른 끼어들기 장면


def scene_parts(scene_id):
    scene, _ = load_scene(RAW_SAMPLES_DIR, scene_id)
    return scene, map_items(scene), {t.id: track_states(t) for t in scene.tracks}


def main():
    configure_plots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = {r["sample_id"]: r for r in read_csv(DATASET_DIR / "labels.csv")}
    metadata = [r | labels[r["sample_id"]] for r in read_csv(DATASET_DIR / "samples.csv") if r["sample_id"] in labels]
    positive = lambda scene_id, target: next(r for r in metadata if r["scenario_id"] == scene_id
                                             and r["target_id"] == target and r["label"] == "1")
    info = json.loads((DATASET_DIR / "dataset_info.json").read_text(encoding="utf-8-sig"))

    scene, items, tracks = scene_parts(EXAMPLE_SCENE)
    plot_cutin_example(OUTPUT_DIR, scene, items, tracks, positive(EXAMPLE_SCENE, EXAMPLE_TARGET))

    scene, items, tracks = scene_parts(MAIN_SCENE)
    main_row = positive(MAIN_SCENE, MAIN_TARGET)
    samples = [r for r in metadata if r["scenario_id"] == MAIN_SCENE]
    with np.load(DATASET_DIR / "inputs.npz", allow_pickle=False) as arrays:
        index = arrays["sample_id"].tolist().index(main_row["sample_id"])
        sequence, tabular, packed = (arrays[k][index] for k in ("X_sequence", "X_basic", "X_bev_packed"))

    plot_raw_scene(OUTPUT_DIR, scene, items, tracks, scene.tracks[scene.sdc_track_index].id)
    plot_raw_contents(OUTPUT_DIR, scene, tracks)
    plot_candidate_pairs(OUTPUT_DIR, scene, items, tracks, main_row)
    plot_anchor_timeline(OUTPUT_DIR, samples)
    plot_label_rule(OUTPUT_DIR, scene, items, tracks, main_row)
    plot_model_inputs(OUTPUT_DIR, sequence, tabular, BASIC_FEATURES)
    plot_bev_input(OUTPUT_DIR, packed)
    plot_dataset_build(OUTPUT_DIR, info)


if __name__ == "__main__":
    main()

