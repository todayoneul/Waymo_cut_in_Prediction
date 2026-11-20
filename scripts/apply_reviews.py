# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scikit-learn==1.9.1"]
# ///
"""검토 Colab에서 내려받은 판정 CSV(data/reviews/*.csv)를 로컬 데이터셋(data/dataset)에 반영한다.

- data/waymo_dataset.zip 원본에서 시작해 data/dataset/을 다시 구성하므로, 보류 판정을 나중에 확정으로 바꿔도 복구된다.
- 판정 규칙은 데이터셋 만들기 노트북 7번과 같다.
  - 끼어들기 있음/없음 → 그 값으로 라벨을 바꾸고 label_source = human
  - 의견 갈림(conflict), 보류(uncertain), 제외(exclude) → 표본을 데이터셋에서 빼고 excluded.csv에 기록
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 프로젝트 폴더의 src 패키지를 불러오기 위해

from src.dataset import SPLITS, read_csv
from src.paths import DATASET_DIR, REVIEWS_DIR

ARCHIVE_PATH = DATASET_DIR.parent / "waymo_dataset.zip"
VALID_STATUS = {"confirmed", "rejected", "uncertain", "exclude"}


def collect_votes(folder):
    votes = defaultdict(dict)
    for path in sorted(folder.glob("*.csv")):
        for row in read_csv(path):
            if row.get("status") in VALID_STATUS:
                votes[row["sample_id"]][row.get("reviewer") or path.stem] = row["status"]
    return votes


def decide(statuses):
    s = set(statuses)
    if "exclude" in s:
        return "exclude"
    if {"confirmed", "rejected"} <= s:
        return "conflict"
    return "confirmed" if "confirmed" in s else "rejected" if "rejected" in s else "uncertain"


def replace_file(path, write):
    with NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path, fields, rows):
    def write(tmp):
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    replace_file(path, write)


def apply_from(source_dir, target_dir, votes):
    labels = read_csv(source_dir / "labels.csv")
    samples = read_csv(source_dir / "samples.csv")
    basic_rows = read_csv(source_dir / "features_basic.csv")
    excluded = read_csv(source_dir / "excluded.csv")
    sample_by_id = {row["sample_id"]: row for row in samples}
    basic_by_id = {row["sample_id"]: row for row in basic_rows}
    gone = {row["sample_id"] for row in excluded}

    keep, summary = [], Counter()
    for row in labels:
        sid = row["sample_id"]
        by_reviewer = votes.get(sid)
        if by_reviewer:
            final = decide(by_reviewer.values())
            summary[final] += 1
            if final in ("confirmed", "rejected"):
                new_label = "1" if final == "confirmed" else "0"
                summary["label_changed"] += row["label"] != new_label
                row.update(label=new_label, label_source="human", review_status=final)
            else:
                auto_label = sample_by_id.get(sid, {}).get("auto_label", "")
                excluded.append({"sample_id": sid, "split": row["split"], "auto_label": auto_label, "reason": f"human_{final}"})
                continue
        keep.append(row)

    known = {row["sample_id"] for row in labels} | gone
    summary["not_in_dataset"] = sum(sid not in known for sid in votes)
    summary["already_excluded"] = sum(sid in gone for sid in votes)

    kept_ids = [row["sample_id"] for row in keep]
    with np.load(source_dir / "inputs.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    index = {sid: i for i, sid in enumerate(arrays["sample_id"].tolist())}
    order = np.array([index[sid] for sid in kept_ids], dtype=int)

    target_dir.mkdir(parents=True, exist_ok=True)
    for extra in ("README.txt", "feature_names.json", "feature_selection.json", "review_queue.csv", "split_plan.json"):
        src_extra = source_dir / extra
        if src_extra.is_file() and src_extra.resolve() != (target_dir / extra).resolve():
            (target_dir / extra).write_bytes(src_extra.read_bytes())

    def write_inputs(tmp):
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **{key: value[order] for key, value in arrays.items()})

    replace_file(target_dir / "inputs.npz", write_inputs)
    write_csv(target_dir / "labels.csv", list(labels[0]), keep)
    write_csv(target_dir / "samples.csv", list(samples[0]), [sample_by_id[sid] for sid in kept_ids])
    write_csv(target_dir / "features_basic.csv", list(basic_rows[0]), [basic_by_id[sid] for sid in kept_ids])
    write_csv(target_dir / "excluded.csv", ["sample_id", "split", "auto_label", "reason"], excluded)

    info = json.loads((source_dir / "dataset_info.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        rows = [row for row in keep if row["split"] == split]
        info["counts"][split].update(samples=len(rows), human_labels=sum(row["label_source"] == "human" for row in rows))
        if split == "train":
            positives = sum(row["label"] == "1" for row in rows)
            info["counts"][split]["positives"] = positives
            info["train_class_weight_for_positive"] = round((len(rows) - positives) / max(positives, 1), 3)
    info["excluded"] = dict(Counter(row["reason"] for row in excluded))
    (target_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return keep, summary


def main():
    votes = collect_votes(REVIEWS_DIR)
    if not votes:
        print(f"{REVIEWS_DIR}에 판정 CSV가 없습니다. 검토 Colab 5번에서 내려받은 CSV를 넣으세요.")
        return
    if ARCHIVE_PATH.is_file():
        with TemporaryDirectory() as tmp:
            with ZipFile(ARCHIVE_PATH) as archive:
                archive.extractall(tmp)
            keep, summary = apply_from(Path(tmp), DATASET_DIR, votes)
        source_note = f"원본 ZIP({ARCHIVE_PATH.name}) 기준 재구성"
    else:
        keep, summary = apply_from(DATASET_DIR, DATASET_DIR, votes)
        source_note = "현재 폴더 기준 반영 (원본 ZIP 없음)"

    print(f"{source_note} · 판정 {sum(len(v) for v in votes.values())}건 (표본 {len(votes)}개, 검토자 {len({r for v in votes.values() for r in v})}명)")
    print(f"라벨 확정 {summary['confirmed'] + summary['rejected']}건 (자동 라벨과 달라진 것 {summary['label_changed']}건), "
          f"제외 {summary['conflict'] + summary['uncertain'] + summary['exclude']}건 (의견 갈림 {summary['conflict']}, 보류 {summary['uncertain']})")
    if summary["already_excluded"] or summary["not_in_dataset"]:
        print(f"이미 제외된 표본 {summary['already_excluded']}건, 데이터셋에 없는 표본 {summary['not_in_dataset']}건은 건너뜀")
    print(f"남은 표본 {len(keep):,}건 → {DATASET_DIR}")


if __name__ == "__main__":
    main()
