# %% [markdown]
# # Part D · 전체 BEV 시계열 생성 (Train · Validation · Test)
# 기존 검토 데이터의 차량 쌍과 기준 시점을 사용해 전체 64개 원본 파일의 1m·0.2m BEV 시계열(과거 1초 · 11프레임)을 생성합니다.
# 원본 파일마다 완성된 ZIP을 Drive에 저장하므로 중간에 끊겨도 다시 실행하면 남은 파일부터 이어서 처리합니다.

# %%
# Drive를 연결하고 입력 데이터셋, 결과 폴더, 처리 분할, 병렬 작업 수를 지정합니다.
import os
from pathlib import Path
from google.colab import drive, auth

drive.mount("/content/drive")
HOME_DRIVE = Path("/content/drive/MyDrive")
DATASET = HOME_DRIVE / "waymo_dataset/dataset"
OUTPUT = HOME_DRIVE / "bev_sequences"
SPLITS = ["train", "validation", "test"]
WORKERS = min(6, max(1, os.cpu_count() or 1))
print(f"CPU 코어 {os.cpu_count()}개 · 동시 처리 {WORKERS}개 · 대상 분할 {SPLITS}")

# %%
# GCS를 읽을 계정으로 인증하고 필요한 패키지 상태를 확인합니다.
import importlib.util, subprocess, sys
if importlib.util.find_spec("google_crc32c") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "google-crc32c>=1.5,<2"])

import google.auth
from google.cloud import storage

auth.authenticate_user()
CREDENTIALS, PROJECT = google.auth.default()
BUCKET = "waymo_open_dataset_motion_v_1_3_1"
print("GCS 인증 완료")

# %% [markdown]
# ## 변환 함수
# 아래 셀은 로컬 Python 소스에서 생성합니다. 원본은 장면 단위로 읽고, 지도는 사례당 한 번, 차량은 11개 시점에 걸쳐 저장합니다. 기준 차량 윤곽은 4.7 × 2.0m로 통일합니다.

# %%
# 원본 레코드의 무결성을 확인하고 공식 Scenario 정의로 해석합니다.
<<BEV_READER>>

# %%
# 기존 프로젝트의 BEV 시계열 변환 함수를 불러옵니다.
<<BEV_MODULE>>

# %%
# 분할 계획과 표본 목록을 한 번에 읽고, 파일 단위 변환·검증 함수를 정의합니다.
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

PLAN = json.loads((DATASET / "split_plan.json").read_text(encoding="utf-8"))
ROWS_BY_SOURCE = rows_by_source(DATASET, splits=SPLITS)
SIGNATURE_FIELDS = ("sample_id", "label", "label_source", "segment_id", "split", "time_target_s")


def archive_ready(path, source, rows):
    if not path.is_file():
        return False
    with ZipFile(path) as archive:
        meta = json.loads(archive.read("manifest.json"))
        signature = lambda records: sorted(tuple(r[k] for k in SIGNATURE_FIELDS) for r in records)
        if (meta["excerpt"] or meta["source"] != source or signature(rows) != signature(meta["samples"])
                or archive.testzip() is not None):
            raise ValueError(f"기존 ZIP과 현재 데이터가 다릅니다. 확인 후 정리하세요: {path}")
    return True


def build_tasks():
    tasks = []
    for key, spec in sorted(PLAN["files"].items()):
        split = spec["split"]
        if split not in SPLITS:
            continue
        source = key.rsplit("/", 1)[-1]
        number = int(source.split("training.tfrecord-")[1].split("-of-")[0])
        rows = ROWS_BY_SOURCE.get(source)
        if not rows:
            raise ValueError(f"표본이 없는 원본 파일입니다: {source}")
        tasks.append((split, number, key, source, int(spec["generation"])))
    return tasks


def process_one(task):
    split, number, key, source, generation = task
    rows = ROWS_BY_SOURCE[source]
    path = OUTPUT / f"{split}_{number:05d}.zip"
    start = time.monotonic()
    last_error = None
    for attempt in range(2):
        try:
            client = storage.Client(project=PROJECT or "", credentials=CREDENTIALS)
            blob = client.bucket(BUCKET).blob(key, generation=generation)
            with blob.open("rb", chunk_size=64 * 1024**2, timeout=300) as stream:
                scenes = (Scenario.FromString(payload) for payload in iter_records(stream))
                count = build_archive(scenes, rows, path, source)
            size_mb = path.stat().st_size / 1024**2
            return path.name, split, count, size_mb, time.monotonic() - start
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(5)
    raise last_error

# %%
# 완료된 파일은 건너뛰고 남은 원본 파일을 병렬로 변환해 Drive에 저장합니다.
OUTPUT.mkdir(parents=True, exist_ok=True)
TASKS = build_tasks()
TODO = []
SKIPPED = 0
for task in TASKS:
    split, number, _, source, _ = task
    path = OUTPUT / f"{split}_{number:05d}.zip"
    if archive_ready(path, source, ROWS_BY_SOURCE[source]):
        SKIPPED += 1
        print(f"[완료 확인] {path.name} ({split}): {len(ROWS_BY_SOURCE[source]):,}건", flush=True)
    else:
        TODO.append(task)

print(f"전체 {len(TASKS)}개 · 완료 {SKIPPED}개 · 남은 작업 {len(TODO)}개 · 동시 처리 {WORKERS}개", flush=True)
done = SKIPPED
total_start = time.monotonic()

if TODO:
    if WORKERS == 1:
        for task in TODO:
            name, split, count, size_mb, elapsed = process_one(task)
            done += 1
            print(f"[{done}/{len(TASKS)}] {name} ({split}): {count:,}건 · {size_mb:.1f} MiB · {elapsed:.0f}초", flush=True)
    else:
        with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(process_one, task): task for task in TODO}
            for future in as_completed(futures):
                name, split, count, size_mb, elapsed = future.result()
                done += 1
                print(f"[{done}/{len(TASKS)}] {name} ({split}): {count:,}건 · {size_mb:.1f} MiB · {elapsed:.0f}초", flush=True)

print(f"전체 작업 완료: {done}/{len(TASKS)}개 · 총 소요 {(time.monotonic() - total_start) / 60:.1f}분 · 저장 위치: {OUTPUT}", flush=True)
