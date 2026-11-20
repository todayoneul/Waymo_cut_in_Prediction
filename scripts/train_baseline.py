# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scikit-learn==1.9.1"]
# ///
"""공통 baseline: 같은 분할에서 입력(기본 8 / 움직임 64 / 선택 목록 / 전체 99)과 기본 ML 모델을 비교한다.

- MODE = "train_cv"   : train 내부 주행 구간(segment_id) 5-fold 교차검증만 실행 (validation·test 비접근)
- MODE = "validation" : train으로 학습하고 validation으로 모델 설정·임계값 선택
- MODE = "final_test" : validation에서 고른 설정·임계값으로 test를 마지막에 1회 평가
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 프로젝트 폴더의 src 패키지를 불러오기 위해

from src.dataset import describe_splits, evaluate, load_dataset, validation_threshold
from src.models import bootstrap_f1, boosting, fit_select
from src.paths import DATASET_DIR, PREDICTIONS_DIR, RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "baseline"
MODE = "train_cv"   # "train_cv" | "validation" | "final_test"


def make_logistic(class_weight=None):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight=class_weight))


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=20,
                                                class_weight="balanced_subsample", random_state=1729, n_jobs=-1))


def detail_metrics(y, scores, threshold, rows):
    """기본 지표에 리드타임 구간별 재현율과 기준 차량 종류별 AP를 함께 계산한다."""
    out = evaluate(y, scores, threshold)
    lead = np.array([float(r["lead_time_s"]) if r["lead_time_s"] else np.nan for r in rows])
    out["recall_by_lead_time"] = {}
    for label, lo, hi in (("0-1s", 0.0, 1.005), ("1-2s", 1.005, 2.005), ("2-3s", 2.005, 9.0)):
        pick = (y == 1) & (lead > lo) & (lead <= hi)
        out["recall_by_lead_time"][label] = {"found": int((scores[pick] >= threshold).sum()), "total": int(pick.sum())}
    out["ap_by_reference_type"] = {}
    for ref_type in ("waymo_sdc", "other_vehicle"):
        sub = np.array([r["reference_type"] == ref_type for r in rows])
        if set(y[sub]) == {0, 1}:
            out["ap_by_reference_type"][ref_type] = float(average_precision_score(y[sub], scores[sub]))
    return out


def run_train_cv(inputs, y, rows, train_mask):
    """train 안에서 주행 구간 단위 5-fold OOF 예측으로 모델·입력별 기준 성능을 비교한다."""
    tr_idx = np.flatnonzero(train_mask)
    y_tr = y[tr_idx]
    rows_tr = [rows[i] for i in tr_idx]
    segments_tr = np.array([r["segment_id"] for r in rows_tr])
    folds = list(GroupKFold(n_splits=5).split(tr_idx, y_tr, groups=segments_tr))

    specs = (
        ("logistic", lambda: make_logistic(None)),
        ("logistic_balanced", lambda: make_logistic("balanced")),
        ("random_forest", make_rf),
        ("boosting", lambda: boosting(0.07, 15, 60, 200, None)),
        ("boosting_balanced", lambda: boosting(0.07, 15, 60, 200, "balanced")),
    )
    results = {}
    oof_scores = {}  # 모델·입력 조합별 교차검증 예측값 (결과 그림용)
    for input_name, x_all in inputs.items():
        x_tr = x_all[tr_idx]
        for model_name, build in specs:
            oof = np.zeros(len(y_tr), dtype=float)
            for fit_i, held_i in folds:
                model = build().fit(x_tr[fit_i], y_tr[fit_i])
                oof[held_i] = model.predict_proba(x_tr[held_i])[:, 1]
            threshold = validation_threshold(y_tr, oof)
            metrics = detail_metrics(y_tr, oof, threshold, rows_tr)
            metrics["f1_95pct_by_segment"] = bootstrap_f1(y_tr, oof, threshold, segments_tr)
            results[f"{model_name} · {input_name}"] = {"columns": x_tr.shape[1], "train_cv_oof": metrics}
            oof_scores[f"{model_name}__{input_name}"] = oof.astype(np.float32)
            print(f"{model_name:18} {input_name:14} CV AP {metrics['average_precision']:.3f} · F1 {metrics['f1']:.3f} "
                  f"(P {metrics['precision']:.3f} / R {metrics['recall']:.3f})", flush=True)
    return results, oof_scores


def run_validation_or_test(inputs, y, rows, train, val, test, eval_test):
    val_rows = [r for r, m in zip(rows, val) if m]
    test_rows = [r for r, m in zip(rows, test) if m]
    val_segments = np.array([r["segment_id"] for r in val_rows])
    test_segments = np.array([r["segment_id"] for r in test_rows])

    # 설정이 고정된 모델: train으로 학습하고 validation 점수만 계산한다.
    fixed_models = (
        ("logistic", "C=1, L2", lambda: make_logistic(None)),
        ("logistic_balanced", "C=1, balanced", lambda: make_logistic("balanced")),
        ("random_forest", "200 trees, balanced_subsample", make_rf),
    )
    # 트리 부스팅: 설정 후보 중 validation AP가 가장 높은 것을 고른다.
    tuned_models = (("boosting", None), ("boosting_balanced", "balanced"))
    results = {}
    for input_name, x in inputs.items():
        candidates = []
        for model_name, config, build in fixed_models:
            model = build().fit(x[train], y[train])
            candidates.append((model_name, config, model, model.predict_proba(x[val])[:, 1]))
        for model_name, class_weight in tuned_models:
            config, model, val_scores = fit_select(x, y, train, val, class_weight=class_weight)
            candidates.append((model_name, config, model, val_scores))
        for model_name, config, model, val_scores in candidates:
            threshold = validation_threshold(y[val], val_scores)
            val_metrics = detail_metrics(y[val], val_scores, threshold, val_rows)
            val_metrics["f1_95pct_by_segment"] = bootstrap_f1(y[val], val_scores, threshold, val_segments)
            item = {"columns": x.shape[1], "config": str(config), "validation": val_metrics}
            if eval_test:
                test_scores = model.predict_proba(x[test])[:, 1]
                test_metrics = detail_metrics(y[test], test_scores, threshold, test_rows)
                test_metrics["f1_95pct_by_segment"] = bootstrap_f1(y[test], test_scores, threshold, test_segments)
                item["test"] = test_metrics
            results[f"{model_name} · {input_name}"] = item
            print(f"{model_name:18} {input_name:14} val AP {val_metrics['average_precision']:.3f} · F1 {val_metrics['f1']:.3f} "
                  f"(P {val_metrics['precision']:.3f} / R {val_metrics['recall']:.3f})", flush=True)
    return results


def main():
    data = load_dataset(DATASET_DIR)
    train, val, test = (data.mask(s) for s in ("train", "validation", "test"))
    inputs = {
        "basic_8": data.basic,                                   # 기본 숫자 8개
        f"motion_{len(data.motion_names)}": data.motion,         # 차량 움직임 전용 (시계열 NN 비교 기준)
        f"selected_{len(data.selected_names)}": data.selected,   # train 교차검증 선택 목록
        f"engineered_{len(data.engineered_names)}": data.engineered,  # 움직임 + 도로·주변차 전체
    }
    if MODE == "train_cv":
        results, oof_scores = run_train_cv(inputs, data.y, data.rows, train)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PREDICTIONS_DIR / "oof_train_cv.npz", sample_id=data.ids[train], label=data.y[train], **oof_scores)
        show = ("train",)
    elif MODE in ("validation", "final_test"):
        results = run_validation_or_test(inputs, data.y, data.rows, train, val, test, MODE == "final_test")
        show = ("train", "validation", "test") if MODE == "final_test" else ("train", "validation")
    else:
        raise ValueError(f"알 수 없는 MODE: {MODE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"mode": MODE, "counts": describe_splits(data, show_labels=show), "models": results}
    path = OUTPUT_DIR / f"metrics_{MODE}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("저장:", path)


if __name__ == "__main__":
    main()
