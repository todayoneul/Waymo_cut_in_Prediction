# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scikit-learn==1.9.1", "matplotlib==3.10.9"]
# ///
"""train만으로 데이터를 탐색한다: 라벨 구성, feature 분포·상관, feature 선택, PCA로 차원을 줄였을 때의 변화.

validation·test는 표본 수만 센다. 모든 비교는 train 안에서 주행 구간 단위 5-fold 교차검증으로 한다.
같은 데이터셋으로 이미 계산한 교차검증 결과가 summary.json에 있으면 다시 계산하지 않고 그림만 새로 그린다.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 프로젝트 폴더의 src 패키지를 불러오기 위해

from src.dataset import describe_splits, load_dataset
from src.models import boosting
from src.paths import DATASET_DIR, FIGURES_DIR, RESULTS_DIR
from src.plots.analysis import (
    plot_class_balance, plot_feature_correlations, plot_feature_distributions, plot_feature_selection, plot_pca,
)
from src.plots.style import configure_plots

RESULTS_OUT = RESULTS_DIR / "data_analysis"   # 표·요약
FIGURES_OUT = FIGURES_DIR / "data_analysis"   # 그림
TREE_SETTINGS = (0.07, 15, 60, 200)           # 비교에서는 feature 표현만 바꾸고 모델 설정은 고정
CORRELATION_LIMIT = 0.95


def write_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# feature마다 결측률, 1·50·99 백분위, 클래스별 중앙값을 기록합니다.
def describe_features(x, y, names):
    table = []
    for name, column in zip(names, x.T):
        observed = column[np.isfinite(column)]
        low, median, high = np.percentile(observed, [1, 50, 99]) if observed.size else [None] * 3
        medians = {f"{tag}_median": float(np.median(v)) if (v := column[(y == label) & np.isfinite(column)]).size else None
                   for label, tag in ((0, "negative"), (1, "positive"))}
        table.append(dict(feature=name, missing_pct=float(100 * np.isnan(column).mean()), p01=low, median=median, p99=high, **medians))
    return table


# 절대 상관계수가 0.95 이상인 두 feature 중 앞쪽 것만 남깁니다.
def correlated_columns(scaled):
    norms = np.linalg.norm(scaled, axis=0)
    unit = np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms > 0)
    corr = unit.T @ unit
    keep = []
    for i in range(scaled.shape[1]):
        if norms[i] > 0 and all(abs(corr[i, j]) < CORRELATION_LIMIT for j in keep):
            keep.append(i)
    return keep


# 한 fold에서 다섯 가지 표현(원본, 표준화, 상관 필터, PCA 90%, PCA 95%)을 만들어 AP를 잽니다. 전처리는 fold의 학습 부분에만 맞춥니다.
def fold_scores(x_fit, y_fit, x_held, y_held):
    prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    fit_scaled, held_scaled = prep.fit_transform(x_fit), prep.transform(x_held)
    pca = PCA(svd_solver="full").fit(fit_scaled)
    cumulative = pca.explained_variance_ratio_.cumsum()
    n90, n95 = (int(np.searchsorted(cumulative, f) + 1) for f in (0.90, 0.95))
    fit_pc, held_pc = pca.transform(fit_scaled), pca.transform(held_scaled)
    keep = correlated_columns(fit_scaled)
    settings = {"Full": (x_fit, x_held), "Imputed control": (fit_scaled, held_scaled),
                "Correlation filter": (x_fit[:, keep], x_held[:, keep]),
                "PCA 90%": (fit_pc[:, :n90], held_pc[:, :n90]), "PCA 95%": (fit_pc[:, :n95], held_pc[:, :n95])}
    out = {}
    for name, (a, b) in settings.items():
        scores = boosting(*TREE_SETTINGS).fit(a, y_fit).predict_proba(b)[:, 1]
        out[name] = (a.shape[1], float(average_precision_score(y_held, scores)))
    return out


def main():
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    configure_plots()
    data = load_dataset(DATASET_DIR)
    train = data.mask("train")
    x, y, names = data.engineered[train], data.y[train], data.engineered_names
    drives = np.array([row["segment_id"] for row in data.rows])[train]   # 교차검증 묶음 = 주행 구간
    counts = describe_splits(data)
    stats = describe_features(x, y, names)
    created = data.info["created_utc"]
    saved = json.loads((RESULTS_OUT / "summary.json").read_text(encoding="utf-8")) if (RESULTS_OUT / "summary.json").exists() else {}

    if saved.get("dataset_created_utc") == created:
        results = saved["train_cv_by_drive_segment"]
        print("같은 데이터셋의 교차검증 결과를 다시 사용합니다.", flush=True)
    else:
        per_fold = []
        for fit, held in GroupKFold(n_splits=min(5, len(set(drives)))).split(x, y, groups=drives):
            per_fold.append(fold_scores(x[fit], y[fit], x[held], y[held]))
            print(f"fold {len(per_fold)}: " + " · ".join(f"{k} {v[1]:.3f}" for k, v in per_fold[-1].items()), flush=True)
        results = {name: {"columns": int(np.mean([f[name][0] for f in per_fold])),
                          "average_precision": float(np.mean([f[name][1] for f in per_fold])),
                          "fold_ap": [f[name][1] for f in per_fold]} for name in per_fold[0]}

    prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    pca = PCA(svd_solver="full").fit(scaled := prep.fit_transform(x))
    report = {"dataset_created_utc": created, "splits": counts,
              "train_columns_with_missing_data": sum(r["missing_pct"] > 0 for r in stats),
              "pc1_and_pc2_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
              "train_cv_by_drive_segment": results,
              "data_usage": "train only; validation/test are counted, never plotted or scored"}
    (RESULTS_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_table(RESULTS_OUT / "feature_summary.csv", stats)

    selection = json.loads((DATASET_DIR / "feature_selection.json").read_text(encoding="utf-8"))
    train_rows = [row for row, keep in zip(data.rows, train) if keep]
    plot_class_balance(y, train_rows, FIGURES_OUT)
    plot_feature_distributions(x, y, names, FIGURES_OUT)
    ranked = plot_feature_selection(selection, data.motion_names, FIGURES_OUT)
    plot_feature_correlations(x, names, ranked, FIGURES_OUT)
    plot_pca(pca.transform(scaled), pca.explained_variance_ratio_, y, results, FIGURES_OUT)
    print("저장:", RESULTS_OUT, FIGURES_OUT, flush=True)


if __name__ == "__main__":
    main()
