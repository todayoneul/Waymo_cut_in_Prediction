"""Colab에서 만든 데이터셋(waymo_dataset.zip을 푼 폴더) 읽기, 정합성 검사, 공통 평가 지표."""
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

SPLITS = ("train", "validation", "test")
# 기본 숫자 8개(X_basic). 모두 기준 시점까지의 관측만으로 계산한 값이다.
BASIC_FEATURES = (
    "forward_m",                # 현재 상대 차량이 기준 차량보다 앞(+)/뒤(-)로 떨어진 거리
    "lateral_m",                # 현재 상대 차량이 기준 차량의 왼쪽(+)/오른쪽(-)으로 떨어진 거리
    "relative_forward_mps",     # 현재 앞뒤 상대속도 (상대 차량 - 기준 차량)
    "relative_lateral_mps",     # 현재 좌우 상대속도 (상대 차량 - 기준 차량)
    "target_forward_1s_m",      # 상대 차량이 지난 1초 동안 앞뒤로 이동한 거리
    "target_lateral_1s_m",      # 상대 차량이 지난 1초 동안 좌우로 이동한 거리
    "reference_speed_mps",      # 기준 차량의 현재 속력
    "target_speed_mps",         # 상대 차량의 현재 속력
)
# samples.csv에서 미래를 보고 만든 열. 분석·다중 작업 정답으로만 쓰고 입력으로 쓰지 않는다.
FUTURE_FIELDS = ("auto_label", "entry_frame", "entry_time_s", "lead_time_s", "entry_lane_id", "entry_forward_center_m")


@dataclass
class Dataset:
    ids: np.ndarray            # 표본 번호
    split: np.ndarray          # train / validation / test
    y: np.ndarray              # 정답 (1 = 3초 안에 끼어들기)
    rows: list                 # 표본별 정보 (samples.csv + 라벨 출처·검토 상태)
    basic: np.ndarray          # N x 8 기본 숫자
    sequence: np.ndarray       # N x 11 x 2 x 7 과거 1초 시계열
    bev_packed: np.ndarray     # N x 5 x 48 x 12 비트 압축 BEV
    engineered: np.ndarray     # N x D engineering feature (움직임 64 + 도로·주변차 35 = 99)
    engineered_names: list
    motion_names: list
    road_names: list
    selected_names: list       # train 교차검증으로 고른 engineering feature
    info: dict

    def mask(self, split):
        return self.split == split

    @property
    def motion(self):
        index = [self.engineered_names.index(name) for name in self.motion_names]
        return self.engineered[:, index]

    @property
    def selected(self):
        index = [self.engineered_names.index(name) for name in self.selected_names]
        return self.engineered[:, index]

    def bev(self, index=slice(None)):
        return np.unpackbits(self.bev_packed[index], axis=-1, count=96)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dataset(data_dir):
    labels = read_csv(data_dir / "labels.csv")
    samples = {row["sample_id"]: row for row in read_csv(data_dir / "samples.csv")}
    names = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))
    info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "inputs.npz", allow_pickle=False) as arrays:
        arrays = {key: arrays[key] for key in arrays.files}

    ids = np.array([row["sample_id"] for row in labels])
    eng_names = list(names["X_engineered"])
    motion_names = list(names.get("X_engineered_motion", eng_names))
    road_names = list(names.get("X_engineered_road", []))
    require(len(set(ids)) == len(ids), "labels.csv에 같은 표본 번호가 두 번 있습니다.")
    require(ids.tolist() == arrays["sample_id"].tolist(), "labels.csv와 inputs.npz의 표본 순서가 다릅니다.")
    require(set(ids) == set(samples), "labels.csv와 samples.csv의 표본이 다릅니다.")
    require(tuple(names["X_basic"]) == BASIC_FEATURES, "기본 숫자 8개의 이름이 예상과 다릅니다.")
    n = len(ids)
    require(arrays["X_basic"].shape == (n, 8) and arrays["X_sequence"].shape == (n, 11, 2, 7), "입력 배열 모양이 다릅니다.")
    require(arrays["X_bev_packed"].shape == (n, 5, 48, 12) and arrays["X_engineered"].shape == (n, len(eng_names)), "입력 배열 모양이 다릅니다.")
    require(np.isfinite(arrays["X_basic"]).all() and np.isfinite(arrays["X_sequence"]).all(), "기본 입력에 결측값이 있습니다.")

    rows = [samples[row["sample_id"]] | row for row in labels]
    split = np.array([row["split"] for row in rows])
    require(set(split) <= set(SPLITS), "알 수 없는 분할 이름이 있습니다.")
    scenes = {s: {row["scenario_id"] for row in rows if row["split"] == s} for s in SPLITS}
    require(not (scenes["train"] & scenes["validation"] or scenes["train"] & scenes["test"] or scenes["validation"] & scenes["test"]),
            "같은 장면이 두 분할에 있습니다.")
    drives = {s: {row["segment_id"] for row in rows if row["split"] == s} for s in SPLITS}
    require(not (drives["train"] & drives["validation"] or drives["train"] & drives["test"] or drives["validation"] & drives["test"]),
            "같은 주행 구간이 두 분할에 있습니다.")
    return Dataset(ids=ids, split=split, y=np.array([int(row["label"]) for row in rows]), rows=rows,
                   basic=arrays["X_basic"].astype(np.float64), sequence=arrays["X_sequence"],
                   bev_packed=arrays["X_bev_packed"], engineered=arrays["X_engineered"].astype(np.float64),
                   engineered_names=eng_names, motion_names=motion_names, road_names=road_names,
                   selected_names=list(names["selected_engineered"]), info=info)


def describe_splits(data, show_labels=("train",)):
    """분할별 표본 수. 라벨 구성(양성 수)은 show_labels에 적은 분할만 보여준다."""
    counts = {}
    for split in SPLITS:
        mask = data.mask(split)
        item = {"samples": int(mask.sum()), "source_files": len({r["source_file"] for r, m in zip(data.rows, mask) if m}),
                "label_sources": dict(Counter(r["label_source"] for r, m in zip(data.rows, mask) if m))}
        if split in show_labels:
            item["positives"] = int(data.y[mask].sum())
        counts[split] = item
    return counts


def deduplicate_split_mask(data, split_mask):
    """같은 분할 안에 입력(X_sequence + X_basic)이 완전히 같은 행이 있으면 첫 행만 남기는 마스크를 돌려준다."""
    keep = np.zeros(len(data.ids), dtype=bool)
    seen = set()
    for i in np.flatnonzero(split_mask):
        key = hashlib.blake2b(data.sequence[i].tobytes() + data.basic[i].tobytes(), digest_size=16).digest()
        if key not in seen:
            seen.add(key)
            keep[i] = True
    return keep


def validation_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    # F1이 같으면 더 높은 임계값을 사용한다.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(y, scores, threshold):
    tn, fp, fn, tp = map(int, confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel())
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp,
    }
