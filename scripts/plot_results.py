# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scikit-learn==1.9.1", "matplotlib==3.10.9"]
# ///
"""기본 ML 베이스라인의 train 교차검증 결과를 그림 4장으로 저장한다.

먼저 scripts/train_baseline.py를 MODE = "train_cv"로 실행해 metrics_train_cv.json과 교차검증 예측값을 만들어야 한다.
validation·test는 읽지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 프로젝트 폴더의 src 패키지를 불러오기 위해

from src.dataset import load_dataset
from src.paths import DATASET_DIR, FIGURES_DIR, PREDICTIONS_DIR, RESULTS_DIR
from src.plots.results import (
    BEST, LEAD_BINS, plot_error_breakdown, plot_error_examples, plot_model_comparison, plot_pr_and_threshold,
)
from src.plots.style import configure_plots

OUTPUT_DIR = FIGURES_DIR / "model_results"


def pick_examples(y, scores, lead, threshold):
    """오류 분석에 보여 줄 표본 네 개: 맞힌 양성, 일찍(2초 이상 전) 놓친 양성, 직전(1초 이내)에 놓친 양성, 가장 강한 잘못 경고."""
    caught = np.flatnonzero((y == 1) & (scores >= threshold) & (lead > 1.005) & (lead <= 2.005))
    early_miss = np.flatnonzero((y == 1) & (scores < threshold) & (lead > 2.005))
    late_miss = np.flatnonzero((y == 1) & (scores < threshold) & (lead <= 1.005))
    false_alarm = np.flatnonzero((y == 0) & (scores >= threshold))
    return [
        ("TP", caught[np.argmax(scores[caught])]),
        ("FN (2 s 이상 전)", early_miss[np.argmin(scores[early_miss])]),
        ("FN (1 s 이내)", late_miss[np.argmin(scores[late_miss])]),
        ("FP", false_alarm[np.argmax(scores[false_alarm])]),
    ]


def main():
    configure_plots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RESULTS_DIR / "baseline" / "metrics_train_cv.json").read_text(encoding="utf-8"))["models"]
    with np.load(PREDICTIONS_DIR / "oof_train_cv.npz", allow_pickle=False) as saved:
        oof = {key: saved[key] for key in saved.files}
    data = load_dataset(DATASET_DIR)
    index = {sample_id: i for i, sample_id in enumerate(data.ids)}
    rows_index = np.array([index[s] for s in oof.pop("sample_id")])
    y = oof.pop("label")
    rows = [data.rows[i] for i in rows_index]

    plot_model_comparison(metrics, OUTPUT_DIR)
    threshold = plot_pr_and_threshold(y, oof, OUTPUT_DIR)
    scores = oof[BEST].astype(float)
    recall = plot_error_breakdown(y, scores, rows, threshold, OUTPUT_DIR)

    lead = np.array([float(r["lead_time_s"]) if r["lead_time_s"] else np.nan for r in rows])
    examples = []
    for name, k in pick_examples(y, scores, lead, threshold):
        timing = f" · Lead time {lead[k]:.1f} s" if y[k] == 1 else " · 실제 Label 0"
        examples.append((f"{name}{timing} · p = {scores[k]:.2f}", data.bev(rows_index[k]), data.sequence[rows_index[k]]))
    plot_error_examples(examples, OUTPUT_DIR)
    summary = {"best": BEST, "threshold": threshold, "recall_by_lead_time_0.5s": recall, "lead_bins": LEAD_BINS[1:-1]}
    (RESULTS_DIR / "baseline" / "figure_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()

