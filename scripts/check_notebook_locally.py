"""데이터셋 노트북을 Colab 없이 로컬에서 끝까지 실행해 봅니다.

data/raw_samples의 원본 3개를 train/validation/test 파일 하나씩으로 쓰고,
Drive·Waymo 로그인 셀 대신 로컬 경로를 넣습니다. 결과는 임시 폴더에 저장됩니다.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from build_notebooks import ROOT, code_cells

RAW = ROOT / "data" / "raw_samples"
SKIP = ("# Colab에 필요한", "# 완성된 ZIP을")   # 설치·다운로드 셀은 로컬에서 필요 없음
# 표본 원본은 train 파일이 1개라 파일 단위 교차검증을 할 수 없어, feature 선택 대신 100개 전부를 씁니다.
SELECTION_CELLS = ("# train 안에서 주행 구간 단위", "# train만으로 feature 선택")


# 로컬 원본 파일을 Colab의 원본 목록(SOURCES)처럼 등록합니다.
def local_sources(g):
    files = sorted(RAW.glob("*.tfrecord"))
    splits = ["train", "validation", "test"]
    g["SOURCES"] = {g["PREFIX"] + path.name: (splits[i], "local") for i, path in enumerate(files)}
    g["SPLIT_FILES"] = {split: [i] for i, split in enumerate(splits)}


# Waymo 형식 정의를 인터넷 대신 저장해 둔 scenario.desc에서 읽습니다.
def local_schema(g):
    pool = descriptor_pool.DescriptorPool()
    for item in descriptor_pb2.FileDescriptorSet.FromString((RAW / "scenario.desc").read_bytes()).file:
        pool.Add(item)
    g["Scenario"] = message_factory.GetMessageClass(pool.FindMessageTypeByName("waymo.open_dataset.Scenario"))


# 원본 파일을 GCS 대신 로컬에서 읽어 파일별 결과를 저장합니다.
def local_processing(g):
    for name, (split, generation) in g["SOURCES"].items():
        start = time.monotonic()
        with open(RAW / Path(name).name, "rb") as stream:
            meta, arrays = g["process_stream"](stream, name, split, generation)
        g["save_npz"](g["result_path"](name), metadata=np.array(json.dumps(meta)), **arrays)
        print(f"{Path(name).name}: 표본 {len(meta['rows'])} · {time.monotonic() - start:.1f}초")


# Waymo 차량 위치 목록도 GCS 대신 로컬 원본에서 모읍니다.
def local_positions(g):
    for name, (split, generation) in g["SOURCES"].items():
        with open(RAW / Path(name).name, "rb") as stream:
            scenes = g["positions_from_stream"](stream)
        g["position_path"](name).write_text(json.dumps({"source": name, "generation": generation, "scenes": scenes}))


def main():
    out = Path(tempfile.mkdtemp(prefix="dataset-check-"))
    g = {}
    for body in code_cells("dataset_build"):
        head = body.strip().splitlines()[0]
        if head.startswith(SKIP):
            continue
        if head.startswith("# Google Drive를 연결"):
            names = ("per_file", "dataset", "reviews", "train_reports")
            g.update(ROOT=out, **{key: out / name for key, name in zip(("PER_FILE_DIR", "DATASET_DIR", "REVIEW_DIR", "REPORT_DIR"), names)})
            for name in names:
                (out / name).mkdir()
        elif head.startswith("# Waymo 이용 권한"):
            local_sources(g)
        elif head.startswith("# Waymo 공식 형식 정의"):
            local_schema(g)
        elif head.startswith("# 아직 처리하지 않은 원본 파일만"):
            exec(body.rsplit("run_all()", 1)[0], g)   # 함수 정의만 실행
            local_processing(g)
        elif head.startswith("# 원본 파일마다 장면별 Waymo 차량 위치"):
            exec(body.rsplit("collect_positions()", 1)[0], g)   # 함수 정의만 실행
            local_positions(g)
        elif head.startswith(SELECTION_CELLS):
            g["SELECTED"] = list(range(len(g["ENGINEERED_NAMES"])))
            g["SELECTION"] = {"selected": g["ENGINEERED_NAMES"]}
        else:
            exec(compile(body, head[:40], "exec"), g)
    print("결과 폴더:", out)


if __name__ == "__main__":
    main()
