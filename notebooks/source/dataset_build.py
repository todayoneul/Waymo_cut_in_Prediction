# %% [markdown]
# # Waymo 끼어들기 예측 데이터셋 만들기
#
# 옆 차로의 상대 차량이 **앞으로 3초 안에 기준 차량 앞으로 끼어들지**를 예측하기 위한 학습 데이터셋을 Waymo 원본에서 만듭니다. 위에서 아래로 한 번 실행하면 **원본 처리 → 라벨 → train/validation/test 분할 → 모델 입력 → feature engineering → feature 선택 → 내보내기**까지 끝납니다.
#
# | 단계 | 하는 일 | 결과 |
# |---|---|---|
# | 1. 준비 | 설치, Drive 연결, Waymo 로그인, 원본 형식 불러오기 | |
# | 2. 분할 계획 고정 | 원본 **파일 단위**로 train/validation/test를 정하고 저장 | \`split_plan.json\` |
# | 3~5. 규칙 정의 | 장면 해석, 후보 차량 쌍, 라벨 규칙, 모델 입력 | |
# | 6. 원본 처리 | 파일마다 표본·입력을 만들어 Drive에 저장 (끊겨도 이어서 실행) | \`per_file/*.npz\` |
# | 7. 데이터셋 조립 | 파일 결과 합치기, 중복·누수 검사, 사람 판정 반영 | |
# | 8. Feature engineering | 정해진 수식으로 99개 feature 계산 | |
# | 9. Feature 선택 | **train 안에서만** 결측·상관·교차검증 중요도로 선택 | \`train_reports/\` |
# | 10. 사람 검증 목록 | 파일별 무작위 검토 배치 만들기 | \`review_queue.csv\` |
# | 11. 저장 | 검사 후 ZIP으로 묶기 | \`waymo_dataset.zip\` |
#
# **지키는 원칙**
#
# 1. **분할은 데이터를 읽기 전에 정합니다.** 같은 원본 파일의 장면은 한 분할에만 들어갑니다. 2번 셀이 저장한 계획과 다르게 바꾸려 하면 실행이 멈춥니다.
# 2. **validation·test는 학습 직전까지 열어 보지 않습니다.** 이 노트북은 두 분할에 대해 표본 수와 데이터 품질 검사 결과만 출력합니다. 분포 그림, feature 중요도, 상관 분석, 모든 설정 선택은 **train만** 사용합니다.
# 3. **미래 정보는 정답을 만들 때만 씁니다.** 모델 입력은 기준 시점의 과거 1초와 현재 정보만으로 만듭니다.
# 4. **결과는 파일마다 저장합니다.** 런타임이 끊기면 같은 셀을 다시 실행해 이어서 진행합니다.

# %% [markdown]
# ### 설계 결정과 이유
#
# | 항목 | 선택 | 이유 |
# |---|---|---|
# | 분할 단위 | **원본 파일 단위** | 같은 원본 파일 안의 장면은 장소·시간대가 겹칠 수 있습니다. 파일째로 나누면 비슷한 장면이 train과 test에 함께 들어가는 일을 줄일 수 있습니다. |
# | 분할 시점 | 데이터를 읽기 **전에** 파일 번호로 고정 (train 0~39번, validation 900~911번, test 950~961번) | 결과를 보고 분할을 바꾸면 평가 점수가 실제보다 좋게 나옵니다. 서로 멀리 떨어진 번호를 써서 분할 사이 겹침도 줄입니다. |
# | 처리 파일 수 | **64개** (train 40, validation 12, test 12) | 파일 하나에서 끼어들기 양성이 약 40건 나옵니다. 같은 주행 구간을 제외한 뒤에도 validation·test에 각 300~400건 이상 양성이 남아 모델 비교가 안정적입니다. |
# | 음성 표본 | **명확한 음성을 모두 저장** | 음성을 얼마나 줄일지는 모델마다 다를 수 있어 학습 단계에서 정합니다. 데이터셋은 실제 비율을 그대로 두고, train 기준 클래스 가중치 값을 함께 제공합니다. |
# | 모델 입력 | 기본 8개 + 과거 1초 시계열 + 위에서 본 지도 이미지 + **engineering 99개** (기준 차량 길이 제외) | 같은 표본을 표 모델, 시계열 모델, CNN이 모두 쓸 수 있게 한 번에 만듭니다. Waymo 차량 고유 길이(5.29 m)를 보고 판별하는 지름길을 막기 위해 기준 차량 크기는 표준값(4.7 m × 2.0 m)으로 고정합니다. |
# | Feature 선택 | **train 안의 교차검증으로만** 선택 | validation·test를 보고 feature를 고르면 평가가 공정하지 않습니다. |
# | 사람 검증 | **선택 사항**, 원본 파일 하나를 한 배치로 검토 | 규칙 라벨만으로 데이터셋이 완성됩니다. 판정이 쌓이면 노트북을 다시 모두 실행해 반영합니다(이미 처리한 원본은 건너뜀). |
# | 같은 주행 구간·중복 | Waymo 차량 위치가 겹치는 장면을 **한 구간으로 묶어** 뒤 분할에서 제외하고, 각 분할 내부의 동일 입력 중복도 제거 | Waymo 장면은 긴 주행 기록에서 잘라낸 9초 구간이라 원본 파일이 달라도 같은 주행이 겹칠 수 있고, 분할 내부 중복도 평가 왜곡을 막기 위해 제거합니다. |
# | 양성 시점 선택 | 진입 시점 기준 가장 가까운 기준 차량을 고르고, 그 차량의 기준 시점 중 **무작위 1개**를 선택 | 가장 이른 시점만 고르면 2~3초 전에 치우치고 Waymo 차량 우선 선택 시 양성 비율 편향이 생기므로, 음성과 마찬가지로 시점을 고르게 뽑습니다. |

# %% [markdown]
# ## 1. 준비

# %%
# Colab에 필요한 라이브러리를 설치합니다.
%pip -q install "google-cloud-storage>=2.16,<4" "grpcio-tools==1.76.0" "protobuf>=6.31.1,<7" "google-crc32c>=1.5,<2" "pillow>=10.1" scipy scikit-learn matplotlib

# %%
# 저장 위치, 분할에 쓸 원본 파일, 라벨 규칙, 병렬 처리 수를 한곳에서 정합니다. 처리를 시작한 뒤에는 바꾸지 않습니다.
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import csv, hashlib, io, json, math, os, re, struct, sys, time

import numpy as np

OUTPUT_NAME = "waymo_dataset"      # Google Drive 안의 결과 폴더 이름
BUCKET = "waymo_open_dataset_motion_v_1_3_1"
PREFIX = "uncompressed/scenario/training/"
SCHEMA_REV = "99a4cb3ff07e2fe06c2ce73da001f850f628e45a"   # Waymo 공식 형식 정의 버전
SEED = 1729                         # 음성 시점 선택과 검토 목록 추출에 쓰는 고정 난수 값

# 원본 파일 번호(training.tfrecord-XXXXX-of-01000)로 분할을 정합니다.
SPLIT_FILES = {
    "train": list(range(0, 40)),          # 앞쪽 40개 파일
    "validation": list(range(900, 912)),  # train과 멀리 떨어진 12개 파일
    "test": list(range(950, 962)),        # validation과도 떨어진 12개 파일
}

# 라벨 규칙. 처리를 시작한 뒤 바꾸면 파일마다 기준이 달라지므로 고정합니다.
RULE = {
    "anchor_stride_frames": 10,   # 기준 시점을 1초(10프레임)마다 이동
    "history_frames": 10,         # 모델 입력: 과거 1초
    "horizon_s": 3.0,             # 정답 확인: 미래 3초
    "dwell_frames": 3,            # 기준 차로 진입 후 3프레임 유지해야 양성
    "reference_radius_m": 75.0,   # 일반 차량을 기준 차량으로 쓸 때 Waymo 차량과의 최대 거리
    "dedup_window_s": 0.75,       # 같은 끼어들기를 하나로 합치는 시간 범위
    "min_reference_speed_mps": 2.0,
    "positive_selection": "closest_reference_uniform_anchor",   # 가장 가까운 기준 차량 + 기준 시점 균등 선택
}
BASIC_FEATURES = ["forward_m", "lateral_m", "relative_forward_mps", "relative_lateral_mps",
                  "target_forward_1s_m", "target_lateral_1s_m", "reference_speed_mps", "target_speed_mps"]
SEQUENCE_FIELDS = ["x_m", "y_m", "vx_mps", "vy_mps", "yaw_rad", "length_m", "width_m"]
BEV_CHANNELS = ["lane_center", "road_lines_and_edges", "reference", "target", "other_vehicles"]
BEV_H, BEV_W = 48, 96             # 1칸 = 1 m. 기준 차량 뒤 24 m ~ 앞 71 m, 좌우 24 m
WORKERS = max(1, os.cpu_count() or 1)

for name, ids in SPLIT_FILES.items():
    print(f"{name:10} 파일 {len(ids):2d}개 · {ids[0]:05d}~{ids[-1]:05d}")
assert not (set(SPLIT_FILES["train"]) & set(SPLIT_FILES["validation"]) |
            set(SPLIT_FILES["train"]) & set(SPLIT_FILES["test"]) |
            set(SPLIT_FILES["validation"]) & set(SPLIT_FILES["test"])), "분할 사이에 같은 파일이 있습니다."

# %%
# Google Drive를 연결하고 결과 폴더를 만듭니다.
from google.colab import drive
drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive") / OUTPUT_NAME
PER_FILE_DIR, DATASET_DIR, REVIEW_DIR, REPORT_DIR = (ROOT / n for n in ("per_file", "dataset", "reviews", "train_reports"))
for folder in (PER_FILE_DIR, DATASET_DIR, REVIEW_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
print("결과 폴더:", ROOT)

# %%
# Waymo 이용 권한이 있는 계정으로 로그인하고, 분할 계획의 원본 파일이 모두 있는지 확인합니다.
from google.colab import auth
from google.cloud import storage
import google.auth

auth.authenticate_user()
CREDENTIALS, PROJECT = google.auth.default()
GCS = storage.Client(project=PROJECT or "", credentials=CREDENTIALS)

def source_name(index):
    return f"{PREFIX}training.tfrecord-{index:05d}-of-01000"

SOURCES = {}   # 원본 경로 → (분할, 파일 버전)
for split, indices in SPLIT_FILES.items():
    for index in indices:
        blob = GCS.bucket(BUCKET).get_blob(source_name(index))
        if blob is None:
            raise RuntimeError(f"원본 파일을 찾지 못했습니다: {source_name(index)}. Waymo 이용 권한을 확인하세요.")
        SOURCES[blob.name] = (split, str(blob.generation))
print(f"원본 파일 {len(SOURCES)}개 확인")

# %%
# Waymo 공식 형식 정의(scenario.proto)를 내려받아 원본 장면을 해석할 준비를 합니다.
from urllib.request import urlopen
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import subprocess

def compile_scenario_schema(revision):
    root_url = f"https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/{revision}/src"
    main_proto = "waymo_open_dataset/protos/scenario.proto"
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pending, seen = [main_proto], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            with urlopen(f"{root_url}/{name}", timeout=30) as response:
                source = response.read()
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
            pending.extend(re.findall(r'^import(?: public| weak)? "([^"]+)";', source.decode(), re.M))
        descriptor = root / "scenario.desc"
        subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{root}", "--include_imports",
                        f"--descriptor_set_out={descriptor}", str(root / main_proto)], check=True)
        return descriptor.read_bytes()

def scenario_class(descriptor):
    pool = descriptor_pool.DescriptorPool()
    for item in descriptor_pb2.FileDescriptorSet.FromString(descriptor).file:
        pool.Add(item)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("waymo.open_dataset.Scenario"))

Scenario = scenario_class(compile_scenario_schema(SCHEMA_REV))

# %%
# 원본 TFRecord 파일을 내려받지 않고 네트워크로 읽으면서 장면을 하나씩 꺼냅니다(손상 검사 포함).
import google_crc32c

def masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

def iter_records(stream):
    while header := stream.read(12):
        if len(header) != 12 or masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
            raise ValueError("TFRecord 헤더가 잘렸거나 손상되었습니다.")
        length = struct.unpack("<Q", header[:8])[0]
        if not 0 < length <= 128 * 1024**2:
            raise ValueError("예상하지 못한 TFRecord 크기입니다.")
        payload, tail = stream.read(length), stream.read(4)
        if len(payload) != length or len(tail) != 4 or masked_crc(payload) != struct.unpack("<I", tail)[0]:
            raise ValueError("TFRecord 장면이 잘렸거나 손상되었습니다.")
        yield payload

# %% [markdown]
# ## 2. 분할 계획 고정
#
# 원본을 읽기 전에 **어느 파일이 어느 분할인지**를 \`split_plan.json\`으로 저장합니다. 이후 실행에서 계획이 달라지면 멈춥니다. 결과를 본 뒤 분할을 바꾸는 실수를 막기 위해서입니다.

# %%
# 분할 계획을 저장하고, 이미 저장된 계획과 다르면 멈춥니다.
PLAN = {
    "dataset": "Waymo Open Motion Dataset v1.3.1 (training files)",
    "schema_revision": SCHEMA_REV, "seed": SEED, "rule": RULE,
    "files": {name: {"split": split, "generation": generation} for name, (split, generation) in sorted(SOURCES.items())},
}
PLAN_PATH = ROOT / "split_plan.json"
if PLAN_PATH.exists():
    if json.loads(PLAN_PATH.read_text()) != PLAN:
        raise RuntimeError("저장된 분할 계획과 다릅니다. 분할·규칙을 바꾸려면 새 OUTPUT_NAME으로 처음부터 만드세요.")
    print("저장된 분할 계획과 일치합니다.")
else:
    PLAN_PATH.write_text(json.dumps(PLAN, indent=2, ensure_ascii=False))
    print("분할 계획을 저장했습니다:", PLAN_PATH)
print(Counter(split for split, _ in SOURCES.values()))

# %% [markdown]
# ## 3. 장면 해석: 차량 궤적과 지도 차로
#
# 차량(Vehicle)만 사용합니다. 각 차량의 위치·방향을 가장 가까운 같은 방향 차로 중심선에 대응시켜(Map Matching), 시점마다 "어느 차로에 있는지"를 구합니다.

# %%
# 지도에서 차로 중심선, 옆 차로 관계, 이어지는 차로(출구) 정보를 읽습니다.
from scipy.spatial import cKDTree

class LaneMap:
    def __init__(self, scene):
        self.lanes = {int(f.id): f.lane for f in scene.map_features
                      if f.HasField("lane") and f.lane.type in (1, 2)}   # 1: 고속도로, 2: 일반 도로 차로
        points, tangents, ids, indices = [], [], [], []
        for lane_id, lane in self.lanes.items():
            p = np.array([[v.x, v.y, v.z] for v in lane.polyline], dtype=float)
            if len(p) < 2:
                continue
            v = np.empty_like(p[:, :2])
            v[0], v[-1] = p[1, :2] - p[0, :2], p[-1, :2] - p[-2, :2]
            v[1:-1] = p[2:, :2] - p[:-2, :2]
            norms = np.linalg.norm(v, axis=1)
            keep = norms > 1e-4
            points.extend(p[keep])
            tangents.extend(v[keep] / norms[keep, None])
            ids.extend([lane_id] * int(keep.sum()))
            indices.extend(np.flatnonzero(keep))
        self.points = np.asarray(points).reshape(-1, 3)
        self.tangents = np.asarray(tangents).reshape(-1, 2)
        self.ids, self.indices = np.asarray(ids, dtype=np.int64), np.asarray(indices, dtype=np.int32)
        self.tree = cKDTree(self.points[:, :2]) if len(self.points) else None
        self.neighbors, self.successors = {}, {}
        for lane_id, lane in self.lanes.items():
            for n in (*lane.left_neighbors, *lane.right_neighbors):
                other = int(n.feature_id)
                a = sorted((int(n.self_start_index), int(n.self_end_index)))
                b = sorted((int(n.neighbor_start_index), int(n.neighbor_end_index)))
                self.neighbors.setdefault((lane_id, other), []).append((a, b))
                self.neighbors.setdefault((other, lane_id), []).append((b, a))

    def adjacent(self, lane_a, index_a, lane_b, index_b):
        """두 차량이 지도상 서로 옆 차로에 있는지 (해당 구간 ±3점 허용)."""
        if lane_a == lane_b:
            return False
        return any(a[0] - 3 <= index_a <= a[1] + 3 and b[0] - 3 <= index_b <= b[1] + 3
                   for a, b in self.neighbors.get((lane_a, lane_b), ()))

    def corridor(self, start):
        """출발 차로와, 그 차로에서 이어지는 차로(최대 6단계)의 묶음."""
        if start not in self.successors:
            seen, frontier = {start}, {start}
            for _ in range(6):
                frontier = {n for i in frontier for n in self.lanes[i].exit_lanes if n in self.lanes and n not in seen}
                seen.update(frontier)
                if not frontier:
                    break
            self.successors[start] = seen
        return self.successors[start]

# %%
# 장면의 차량 상태를 배열로 바꾸고, 모든 차량·시점을 가장 알맞은 차로에 대응시킵니다.
def scene_arrays(scene):
    tracks = [tr for tr in scene.tracks if tr.object_type == 1]   # 차량만
    ids = np.asarray([tr.id for tr in tracks], dtype=np.int64)
    times = np.asarray(scene.timestamps_seconds, dtype=float)
    states = np.zeros((len(tracks), len(times), 6), dtype=float)  # x, y, z, 방향, vx, vy
    valid = np.zeros(states.shape[:2], dtype=bool)
    for i, track in enumerate(tracks):
        for k, s in enumerate(track.states[:len(times)]):
            if s.valid:
                values = (s.center_x, s.center_y, s.center_z, s.heading, s.velocity_x, s.velocity_y)
                if all(math.isfinite(v) for v in values):
                    states[i, k] = values
                    valid[i, k] = True
    return ids, times, states, valid

def match_states(lanes, states, valid):
    lane_id = np.full(valid.shape, -1, dtype=np.int64)
    point_index = np.full(valid.shape, -1, dtype=np.int32)
    if lanes.tree is None or not valid.any():
        return lane_id, point_index
    obs = states[valid]
    k = min(24, len(lanes.points))
    dist, nearest = lanes.tree.query(obs[:, :2], k=list(range(1, k + 1)), distance_upper_bound=2.6)
    safe = np.minimum(nearest, len(lanes.points) - 1)
    heading = np.stack([np.cos(obs[:, 3]), np.sin(obs[:, 3])], axis=1)
    align = np.einsum("ni,nki->nk", heading, lanes.tangents[safe])
    height = np.abs(obs[:, None, 2] - lanes.points[safe, 2])
    # 2.6 m 이내, 진행 방향이 비슷하고(cos ≥ 0.64), 높이 차 3 m 이내인 차로 점 중 가장 가까운 점
    cost = np.where((align >= 0.64) & (height <= 3.0), dist + 0.65 * (1 - align), np.inf)
    best = np.argmin(cost, axis=1)
    r = np.arange(len(obs))
    ok = np.isfinite(cost[r, best])
    selected = safe[r, best]
    lane_id[valid] = np.where(ok, lanes.ids[selected], -1)
    point_index[valid] = np.where(ok, lanes.indices[selected], -1)
    return lane_id, point_index

# %% [markdown]
# ## 4. 후보 차량 쌍과 라벨 규칙
#
# | 구분 | 조건 |
# |---|---|
# | **후보** | 기준 차량: 초속 2 m 이상, 과거 1초 관측, 차로 대응 성공. 상대 차량: 같은 방향(cos ≥ 0.70), 지도상 옆 차로, 앞뒤 -15~55 m, 좌우 2.3~11 m, 높이 차 3 m 이내 |
# | **양성 (1)** | 3초 안에 기준 차량과 같은 차로(기준 차로에서 이어지는 차로 포함)로 진입, 기준 차량 앞 0~40 m, 진행 방향 유사, **3프레임 연속 유지**. 원래부터 같은 출구로 이어지는 단순 합류는 제외 |
# | **음성 (0)** | 미래 3초 내내 두 차량이 관측되고, 각자 자기 차로 묶음 안에만 머묾 (명확한 음성) |
# | **보류** | 양성도 명확한 음성도 아닌 경우(관측 끊김, 차로 판정 애매 등). 데이터셋에서 제외 |
#
# 기준 시점은 1초마다 옮깁니다. 같은 끼어들기가 여러 기준 시점에서 잡히면 0.75초 안의 진입을 한 사건으로 묶고, 진입 시점에 가장 가까이 뒤에 있던 기준 차량의 기준 시점 중 하나를 고정 해시로 고르게 뽑습니다. 음성도 차량 쌍마다 기준 시점 하나만 고르게 남깁니다.

# %%
# 기준 시점마다 기준 차량과 옆 차로의 상대 차량 후보 쌍을 찾습니다.
def anchor_pairs(states, history_ok, matches, points, lanes, anchor, waymo_index):
    now = states[:, anchor]
    usable = history_ok & (matches[:, anchor] != -1) & (np.hypot(now[:, 4], now[:, 5]) >= RULE["min_reference_speed_mps"])
    near_waymo = np.hypot(*(now[:, :2] - now[waymo_index, :2]).T) <= RULE["reference_radius_m"]
    for ref in np.flatnonzero(usable):
        if ref != waymo_index and not near_waymo[ref]:
            continue
        direction = np.array([math.cos(now[ref, 3]), math.sin(now[ref, 3])])
        delta = now[:, :3] - now[ref, :3]
        ahead = delta[:, :2] @ direction
        side = delta[:, :2] @ [-direction[1], direction[0]]
        eligible = (usable & (ahead >= -15) & (ahead <= 55) & (np.abs(side) >= 2.3) & (np.abs(side) <= 11) &
                    (np.abs(delta[:, 2]) <= 3.0) & (np.cos(now[:, 3] - now[ref, 3]) >= .70))
        a = int(matches[ref, anchor])
        for target in np.flatnonzero(eligible):
            b = int(matches[target, anchor])
            if lanes.adjacent(a, points[ref, anchor], b, points[target, anchor]):
                yield int(ref), int(target), a, b

# %%
# 미래 3초 안에 상대 차량이 기준 차로 앞쪽으로 들어와 3프레임 유지하는 첫 시점을 찾습니다(양성 판정).
def first_entry(times, states, valid, matches, lanes, ref, target, a, b, t):
    dwell = RULE["dwell_frames"]
    limit = min(len(times) - dwell + 1, int(np.searchsorted(times, times[t] + RULE["horizon_s"] + 0.005, side="right")))
    end = min(len(times), limit + dwell - 1)
    if limit <= t + 1:
        return None
    r, s = states[ref, t + 1:end], states[target, t + 1:end]
    rm, sm = matches[ref, t + 1:end], matches[target, t + 1:end]
    delta = s[:, :2] - r[:, :2]
    gap = delta[:, 0] * np.cos(r[:, 3]) + delta[:, 1] * np.sin(r[:, 3])   # 기준 차량 진행 방향 앞뒤 거리
    qualifies = (valid[ref, t + 1:end] & valid[target, t + 1:end] & (rm != -1) & (rm == sm) &
                 np.isin(rm, list(lanes.corridor(a))) & ~np.isin(sm, list(lanes.corridor(b))) &
                 (np.cos(r[:, 3] - s[:, 3]) >= 0.65) & (gap > 0) & (gap <= 40.0))
    for offset in np.flatnonzero(qualifies[:limit - t - 1]):
        if qualifies[offset:offset + dwell].all():
            return int(t + 1 + offset), int(rm[offset]), float(gap[offset])
    return None

def clean_negative(times, valid, matches, lanes, ref, target, a, b, t):
    """미래 3초 내내 두 차량이 관측되고 각자 자기 차로 묶음에만 머물면 명확한 음성."""
    horizon_end = times[t] + RULE["horizon_s"]
    last = int(np.searchsorted(times, horizon_end + 0.005, side="right")) - 1
    if last <= t or times[last] < horizon_end - 0.005:
        return False
    future = slice(t + 1, last + 1)
    if not (valid[ref, future].all() and valid[target, future].all()):
        return False
    r, s = set(matches[ref, future]), set(matches[target, future])
    ref_route, target_route = lanes.corridor(a), lanes.corridor(b)
    return r <= ref_route and s <= target_route and not (s & ref_route) and not (r & target_route)

# %%
# 같은 끼어들기를 하나로 합치고, 장면 하나에서 양성·음성 표본을 모읍니다.
def stable_number(key):
    """같은 입력이면 항상 같은 큰 정수 (무작위 선택을 재현 가능하게)."""
    return int.from_bytes(hashlib.blake2b(f"{SEED}:{key}".encode(), digest_size=8).digest(), "big")

def merge_same_maneuver(hits):
    by_target, merged = defaultdict(list), []
    for h in hits:
        by_target[h["target_id"]].append(h)
    for hits_for_vehicle in by_target.values():
        groups = []
        for h in sorted(hits_for_vehicle, key=lambda h: h["entry_time_s"]):
            if not groups or h["entry_time_s"] - groups[-1][0]["entry_time_s"] > RULE["dedup_window_s"]:
                groups.append([])
            groups[-1].append(h)
        for group in groups:
            # 진입 시점에 가장 가까이 뒤에 있던 기준 차량을 고르고(Waymo 우선 없음), 그 차량의 기준 시점 중 하나를 고르게 선택합니다.
            closest_ref = min(group, key=lambda h: (h["entry_forward_center_m"], h["reference_id"]))["reference_id"]
            ref_hits = [h for h in group if h["reference_id"] == closest_ref]
            row = dict(min(ref_hits, key=lambda h: stable_number(f"pos:{h['scenario_id']}:{h['reference_id']}:{h['target_id']}:{h['anchor_frame']}")))
            row["sample_id"] = f'{row["scenario_id"]}:{row["reference_id"]}:{row["target_id"]}:{row["anchor_frame"]}'
            row["matches_in_scene"] = len(group)
            merged.append(row)
    return sorted(merged, key=lambda row: (row["entry_frame"], row["target_id"]))

def label_scene(scene):
    """한 장면의 양성(병합 후)과 명확한 음성(차량 쌍마다 한 시점) 표본, 처리 통계를 돌려줍니다."""
    counts = Counter(scenarios_read=1)
    if (not scene.scenario_id or not scene.HasField("current_time_index") or not scene.HasField("sdc_track_index")
            or not 0 <= scene.sdc_track_index < len(scene.tracks)):
        return [], None, counts
    ids, times, states, valid = scene_arrays(scene)
    if len(ids) < 2 or len(times) < 2 or not np.allclose(np.diff(times), 0.1, atol=0.005):
        return [], None, counts
    waymo = np.flatnonzero(ids == scene.tracks[scene.sdc_track_index].id)
    if len(waymo) != 1:
        return [], None, counts
    waymo = int(waymo[0])
    lanes = LaneMap(scene)
    matches, points = match_states(lanes, states, valid)
    positives, negatives = [], {}
    history = RULE["history_frames"]
    horizon = RULE["horizon_s"] + 0.1 * (RULE["dwell_frames"] - 1)
    for t in range(max(history, scene.current_time_index), len(times), RULE["anchor_stride_frames"]):
        if times[-1] - times[t] < horizon - 0.005:
            break
        if not valid[waymo, t]:
            continue
        observed = valid[:, t - history:t + 1].all(axis=1)
        for ref, target, a, b in anchor_pairs(states, observed, matches, points, lanes, t, waymo):
            counts["candidate_windows"] += 1
            row = dict(scenario_id=scene.scenario_id, reference_id=int(ids[ref]), target_id=int(ids[target]),
                       reference_type="waymo_sdc" if ref == waymo else "other_vehicle", anchor_frame=t,
                       reference_lane_id=a, target_lane_id=b, _ref=ref, _target=target)
            event = first_entry(times, states, valid, matches, lanes, ref, target, a, b, t)
            if event is not None:
                frame, lane, gap = event
                row.update(auto_label=1, entry_frame=frame, entry_lane_id=lane, entry_forward_center_m=round(gap, 3),
                           entry_time_s=round(float(times[frame] - times[0]), 4), lead_time_s=round(float(times[frame] - times[t]), 4))
                positives.append(row)
            elif clean_negative(times, valid, matches, lanes, ref, target, a, b, t):
                row.update(auto_label=0, entry_frame="", entry_lane_id="", entry_forward_center_m="", entry_time_s="",
                           lead_time_s="", matches_in_scene=1, sample_id=f"{scene.scenario_id}:{ids[ref]}:{ids[target]}:{t}")
                pair = (ref, target)
                if pair not in negatives or stable_number(row["sample_id"]) < stable_number(negatives[pair]["sample_id"]):
                    negatives[pair] = row
            else:
                counts["uncertain_windows"] += 1
    positives = merge_same_maneuver(positives)
    counts.update(candidate_scenes=int(counts["candidate_windows"] > 0),
                  labeled_scenes=int(bool(positives or negatives)),
                  positive_events=len(positives), clean_negative_pairs=len(negatives))
    return positives + list(negatives.values()), (scene, ids, states, valid), counts

# %% [markdown]
# ## 5. 모델 입력 만들기 (과거 1초 + 현재만 사용)
#
# 모든 좌표는 **기준 시점의 기준 차량을 원점**, 진행 방향을 +x로 회전한 값입니다.
#
# | 입력 | 모양 | 내용 |
# |---|---|---|
# | \`X_basic\` | 8 | 앞뒤·좌우 거리, 앞뒤·좌우 상대속도, 상대 차량의 지난 1초 앞뒤·좌우 이동, 두 차량 속력 |
# | \`X_sequence\` | 11 × 2 × 7 | 과거 1초 11프레임 × (기준, 상대) × (x, y, vx, vy, 방향, 길이, 너비) |
# | \`X_bev_packed\` | 5 × 48 × 12 | 기준 시점 BEV 5채널을 비트로 압축(풀면 5 × 48 × 96) |

# %%
# 기본 숫자 8개와 과거 11프레임 시계열을 기준 시점까지의 관측만으로 계산합니다.
def rotation(heading):
    c, s = math.cos(heading), math.sin(heading)
    return np.array([[c, -s], [s, c]])

def basic_features(states, t, ref, target):
    r, s = states[ref, t], states[target, t]
    c, z = math.cos(r[3]), math.sin(r[3])
    rotate = lambda v: [c * v[0] + z * v[1], -z * v[0] + c * v[1]]
    delta = rotate(s[:2] - r[:2])                                      # 현재 앞뒤·좌우 거리
    velocity = rotate(s[4:6] - r[4:6])                                 # 현재 앞뒤·좌우 상대속도
    movement = rotate(s[:2] - states[target, t - RULE["history_frames"], :2])   # 상대 차량의 지난 1초 이동
    values = delta + velocity + movement + [math.hypot(*r[4:6]), math.hypot(*s[4:6])]
    return [round(float(v), 4) for v in values]

def vehicle_dimensions(scene, ids, n_frames):
    by_id = {tr.id: tr for tr in scene.tracks if tr.object_type == 1}
    dims = np.ones((len(ids), n_frames, 2), dtype=np.float32)
    for i, tid in enumerate(ids):
        for j, s in enumerate(by_id[int(tid)].states[:n_frames]):
            if s.valid:
                dims[i, j] = [max(1., s.length) if math.isfinite(s.length) else 1.,
                              max(.5, s.width) if math.isfinite(s.width) else .5]
    return dims

def sequence_input(states, dims, t, ref, target):
    r = states[ref, t]
    window = slice(t - RULE["history_frames"], t + 1)
    values = states[[ref, target], window].transpose(1, 0, 2)
    rot = rotation(r[3])
    xy = (values[:, :, :2] - r[:2]) @ rot
    v = values[:, :, 4:6] @ rot
    yaw = (values[:, :, 3] - r[3] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate((xy, v, yaw[..., None], dims[[ref, target], window].transpose(1, 0, 2)), axis=2).astype(np.float32)

# %%
# 기준 시점의 지도와 차량을 위에서 본 5채널 격자(BEV)로 그립니다.
from PIL import Image, ImageDraw

def map_lines(scene):
    out = []
    for f in scene.map_features:
        kind = f.WhichOneof("feature_data")
        if kind in ("lane", "road_line", "road_edge"):
            p = np.array([[pt.x, pt.y, pt.z] for pt in getattr(f, kind).polyline])
            if len(p) >= 2:
                out.append((kind, p, float(p[:, 0].min()), float(p[:, 0].max()),
                            float(p[:, 1].min()), float(p[:, 1].max()),
                            float(p[:, 2].min()), float(p[:, 2].max())))
    return out

def bev_pixels(xy):
    return np.rint(np.column_stack((xy[:, 0] + 24., 24. - xy[:, 1]))).astype(int)

def map_layers(lines, r):
    layers = [Image.new("1", (BEV_W, BEV_H)) for _ in range(2)]
    draw = [ImageDraw.Draw(img) for img in layers]
    rot = rotation(r[3])
    rx, ry, rz = r[0], r[1], r[2]
    for kind, p, xmin, xmax, ymin, ymax, zmin, zmax in lines:
        if zmax < rz - 3.0 or zmin > rz + 3.0 or xmax < rx - 80.0 or xmin > rx + 80.0 or ymax < ry - 80.0 or ymin > ry + 80.0:
            continue
        if np.min(np.abs(p[:, 2] - rz)) > 3.0:     # 다른 높이(고가도로 등) 제외
            continue
        q = (p[:, :2] - r[:2]) @ rot
        if q[:, 0].max() < -25 or q[:, 0].min() > 73 or q[:, 1].max() < -25 or q[:, 1].min() > 25:
            continue
        good = np.abs(p[:, 2] - rz) <= 3.0
        starts = np.flatnonzero(good & ~np.r_[False, good[:-1]])
        stops = np.flatnonzero(good & ~np.r_[good[1:], False]) + 1
        target_draw = draw[0 if kind == "lane" else 1]
        for a, b in zip(starts, stops):
            if b - a >= 2:
                target_draw.line([tuple(v) for v in bev_pixels(q[a:b])], fill=1, width=1)
    return np.stack([np.asarray(img) for img in layers])

def bev_input(lines, states, valid, dims, t, ref, target, cache):
    if (t, ref) not in cache:
        r = states[ref, t]
        rot = rotation(r[3])
        polys = {}
        for i in np.flatnonzero(valid[:, t]):
            s = states[i, t]
            d = (s[:2] - r[:2]) @ rot
            if -28 <= d[0] <= 76 and abs(d[1]) <= 28 and abs(s[2] - r[2]) <= 3:
                l, w = dims[i, t] / 2
                corners = np.array([[-l, -w], [l, -w], [l, w], [-l, w]]) @ rotation(r[3] - s[3]) + d
                polys[int(i)] = [tuple(v) for v in bev_pixels(corners)]
        ref_img = Image.new("1", (BEV_W, BEV_H))
        if ref in polys:
            ImageDraw.Draw(ref_img).polygon(polys[ref], fill=1)
        cache[t, ref] = (map_layers(lines, r), np.asarray(ref_img)[None], polys)
    maps, ref_arr, polys = cache[t, ref]
    tgt_img, oth_img = Image.new("1", (BEV_W, BEV_H)), Image.new("1", (BEV_W, BEV_H))
    if target in polys:
        ImageDraw.Draw(tgt_img).polygon(polys[target], fill=1)
    oth_draw = ImageDraw.Draw(oth_img)
    for i, poly in polys.items():
        if i != ref and i != target:
            oth_draw.polygon(poly, fill=1)
    layers = np.concatenate([maps, ref_arr, np.asarray(tgt_img)[None], np.asarray(oth_img)[None]], axis=0)
    return np.packbits(layers, axis=-1)

# %% [markdown]
# ## 6. 원본 파일 처리
#
# 파일 하나를 끝까지 읽어 표본과 입력을 만들고 \`per_file/\`에 저장합니다. 저장된 파일은 다시 처리하지 않으므로, **런타임이 끊기면 1번 준비 셀들과 이 셀을 다시 실행**하면 이어서 진행합니다. CPU 수만큼 파일을 동시에 처리합니다.
#
# validation·test 파일은 처리한 장면·표본 **수만** 출력합니다.

# %%
# 원본 파일 하나를 처리해 표본 정보, 기본 숫자, 시계열, BEV를 파일별 결과로 저장합니다.
META_FIELDS = ["sample_id", "scenario_id", "reference_id", "target_id", "reference_type", "anchor_frame",
               "auto_label", "entry_frame", "entry_time_s", "lead_time_s", "entry_lane_id", "entry_forward_center_m",
               "reference_lane_id", "target_lane_id", "matches_in_scene"]

def result_path(name):
    return PER_FILE_DIR / (Path(name).name + ".npz")

def process_stream(stream, name, split, generation):
    rows, basic, seqs, bevs, counts, seen = [], [], [], [], Counter(), set()
    for payload in iter_records(stream):
        scene = Scenario.FromString(payload)
        if not scene.scenario_id or scene.scenario_id in seen:
            raise ValueError(f"장면 ID가 비어 있거나 한 파일에서 중복됩니다: {name}")
        seen.add(scene.scenario_id)
        samples, observed, scene_counts = label_scene(scene)
        counts.update(scene_counts)
        if not samples:
            continue
        scene, ids, states, valid = observed
        dims, lines, cache = vehicle_dimensions(scene, ids, states.shape[1]), map_lines(scene), {}
        for row in samples:
            t, ref, target = row["anchor_frame"], row["_ref"], row["_target"]
            basic.append(basic_features(states, t, ref, target))
            seqs.append(sequence_input(states, dims, t, ref, target))
            bevs.append(bev_input(lines, states, valid, dims, t, ref, target, cache))
            rows.append({k: row.get(k, "") for k in META_FIELDS} | {"source_file": Path(name).name, "split": split})
    n = len(rows)
    meta = dict(source=name, generation=generation, split=split, rule=RULE, counts=dict(counts), rows=rows)
    return meta, dict(X_basic=np.asarray(basic, np.float32).reshape(n, 8),
                      X_sequence=np.asarray(seqs, np.float32).reshape(n, 11, 2, 7),
                      X_bev_packed=np.asarray(bevs, np.uint8).reshape(n, 5, BEV_H, BEV_W // 8))

def save_npz(path, **arrays):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def finished(name):
    path = result_path(name)
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
    return meta["source"] == name and meta["generation"] == SOURCES[name][1] and meta.get("rule") == RULE

def process_file(name):
    split, generation = SOURCES[name]
    start = time.monotonic()
    client = storage.Client(project=PROJECT or "", credentials=CREDENTIALS)   # 프로세스마다 새 연결
    blob = client.bucket(BUCKET).blob(name, generation=int(generation))
    with blob.open("rb", chunk_size=64 * 1024**2, timeout=300) as stream:
        meta, arrays = process_stream(stream, name, split, generation)
    meta["elapsed_s"] = round(time.monotonic() - start, 1)
    save_npz(result_path(name), metadata=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    return name, split, meta["counts"], len(meta["rows"]), meta["elapsed_s"]

# %%
# 아직 처리하지 않은 원본 파일만 병렬로 처리합니다. 끊기면 이 셀을 다시 실행하면 이어서 합니다.
import multiprocessing as mp

def run_all():
    todo = [name for name in SOURCES if not finished(name)]
    print(f"처리할 파일 {len(todo)}개 / 전체 {len(SOURCES)}개 · 동시 처리 {WORKERS}개", flush=True)
    done = len(SOURCES) - len(todo)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(process_file, name): name for name in todo}
        for future in as_completed(futures):
            name, split, counts, n, elapsed = future.result()
            done += 1
            if split == "train":
                detail = f"양성 {counts.get('positive_events', 0)} · 음성 {counts.get('clean_negative_pairs', 0)}"
            else:
                detail = f"표본 {n:,}건"   # validation·test는 라벨 구성을 출력하지 않습니다.
            print(f"[{done}/{len(SOURCES)}] {Path(name).name} ({split}) 장면 {counts.get('scenarios_read', 0)} · {detail} · {elapsed:.0f}초", flush=True)

run_all()

# %% [markdown]
# ### 같은 주행 구간 찾기
#
# Waymo 장면(9초)은 더 긴 주행 기록에서 잘라낸 구간이라, **장면 ID와 원본 파일이 달라도 같은 주행을 담고 있을 수 있습니다.** 이런 장면이 train과 validation·test에 나뉘면 모델이 이미 본 주행으로 평가하게 됩니다. 장면마다 Waymo 차량이 지나간 위치를 1 cm 단위로 모아 두고, 7번에서 위치가 3곳 이상 같은 장면을 같은 주행 구간으로 묶습니다.

# %%
# 원본 파일마다 장면별 Waymo 차량 위치를 저장합니다. 이미 저장한 파일은 건너뜁니다(파일당 약 30초).
POSITION_DIR = PER_FILE_DIR / "waymo_vehicle_positions"
POSITION_DIR.mkdir(exist_ok=True)

def position_path(name):
    return POSITION_DIR / (Path(name).name + ".json")

def positions_from_stream(stream):
    scenes = {}
    for payload in iter_records(stream):
        scene = Scenario.FromString(payload)
        track = scene.tracks[scene.sdc_track_index]
        scenes[scene.scenario_id] = sorted({f"{s.center_x:.2f},{s.center_y:.2f}" for s in track.states if s.valid})
    return scenes

def save_positions(name):
    client = storage.Client(project=PROJECT or "", credentials=CREDENTIALS)
    blob = client.bucket(BUCKET).blob(name, generation=int(SOURCES[name][1]))
    with blob.open("rb", chunk_size=64 * 1024**2, timeout=300) as stream:
        scenes = positions_from_stream(stream)
    tmp = position_path(name).with_suffix(".tmp")
    tmp.write_text(json.dumps({"source": name, "generation": SOURCES[name][1], "scenes": scenes}))
    os.replace(tmp, position_path(name))
    return name

def collect_positions():
    todo = [name for name in SOURCES if not position_path(name).is_file()]
    print(f"위치를 모을 파일 {len(todo)}개 / 전체 {len(SOURCES)}개", flush=True)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as pool:
        for done, future in enumerate(as_completed([pool.submit(save_positions, name) for name in todo]), 1):
            print(f"[{done}/{len(todo)}] {Path(future.result()).name}", flush=True)

collect_positions()

# %% [markdown]
# ## 7. 데이터셋 조립과 검사
#
# 파일별 결과를 합친 뒤 다음을 확인합니다.
#
# - 표본 번호가 겹치지 않는지, 같은 장면이 두 분할에 나뉘지 않는지
# - 같은 주행 구간이 두 분할에 나뉘는지 (있으면 뒤 분할의 장면을 통째로 제외: train과 같으면 validation·test에서, validation과 같으면 test에서)
# - 그래도 입력이 완전히 같은 표본이 다른 분할에 남아 있는지(뒤 분할 쪽 제외), 같은 분할 안에 완전 중복 표본이 있는지(첫 표본만 유지)
# - 사람 판정 반영: \`reviews/\` 폴더의 판정 CSV와 검토 Colab이 저장한 판정. 의견이 갈리거나 보류인 표본은 제외

# %%
# 파일별 결과를 하나로 합치고, 번호 중복과 분할 사이 장면 겹침을 검사합니다.
def load_results():
    metas, parts = [], defaultdict(list)
    for name in sorted(SOURCES):
        if not finished(name):
            raise RuntimeError(f"아직 처리되지 않은 파일이 있습니다: {name}. 6번 셀을 먼저 끝까지 실행하세요.")
        with np.load(result_path(name), allow_pickle=False) as data:
            metas.append(json.loads(data["metadata"].item()))
            for key in ("X_basic", "X_sequence", "X_bev_packed"):
                parts[key].append(data[key])
    rows = [row for meta in metas for row in meta["rows"]]
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    return metas, rows, arrays

METAS, ROWS, ARRAYS = load_results()
COVERAGE = defaultdict(Counter)
for meta in METAS:
    COVERAGE[meta["split"]].update(meta["counts"])

ids = [r["sample_id"] for r in ROWS]
assert len(ids) == len(set(ids)), "표본 번호가 중복됩니다."
scene_split = defaultdict(set)
for r in ROWS:
    scene_split[r["scenario_id"]].add(r["split"])
crossing = [s for s, splits in scene_split.items() if len(splits) > 1]
assert not crossing, f"두 분할에 나뉜 장면이 있습니다: {crossing[:5]}"
assert np.isfinite(ARRAYS["X_basic"]).all() and np.isfinite(ARRAYS["X_sequence"]).all()
for split in ("train", "validation", "test"):
    c = COVERAGE[split]
    print(f"{split:10} 장면 {c['scenarios_read']:6,} · 라벨이 있는 장면 {c['labeled_scenes']:5,} · 표본 {sum(r['split'] == split for r in ROWS):7,}")

# %%
# Waymo 차량 위치가 3곳 이상 같은 장면을 같은 주행 구간으로 묶고, 표본마다 구간 번호(segment_id)를 붙입니다.
def drive_segments(min_shared=3):
    scene_split, scenes_at = {}, defaultdict(list)
    for name, (split, _) in SOURCES.items():
        for scene_id, points in json.loads(position_path(name).read_text())["scenes"].items():
            scene_split[scene_id] = split
            for point in points:
                scenes_at[point].append(scene_id)
    shared = Counter()
    for scenes in scenes_at.values():
        if 1 < len(scenes) <= 50:
            for a in range(len(scenes)):
                for b in range(a + 1, len(scenes)):
                    shared[tuple(sorted((scenes[a], scenes[b])))] += 1
    parent = {s: s for s in scene_split}
    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    for (a, b), n in shared.items():
        if n >= min_shared:
            parent[max(root(a), root(b))] = min(root(a), root(b))
    return scene_split, {s: root(s) for s in scene_split}

SCENE_SPLIT, SEGMENT = drive_segments()
SPLITS_IN_SEGMENT = defaultdict(set)
for scene_id, segment in SEGMENT.items():
    SPLITS_IN_SEGMENT[segment].add(SCENE_SPLIT[scene_id])
for row in ROWS:
    row["segment_id"] = SEGMENT[row["scenario_id"]]
sizes = Counter(SEGMENT.values())
print(f"장면 {len(SEGMENT):,}개 → 주행 구간 {len(sizes):,}개 (장면 2개 이상인 구간 {sum(n > 1 for n in sizes.values()):,}개)")
print(f"두 분할 이상에 걸친 구간 {sum(len(s) > 1 for s in SPLITS_IN_SEGMENT.values()):,}개")

# %%
# 앞 분할과 같은 주행 구간인 장면, 그리고 입력이 완전히 같은 표본을 뒤 분할에서 제외합니다(학습 데이터 누수 방지).
EXCLUDED = []          # (행 번호, 이유)
for i, row in enumerate(ROWS):
    earlier = SPLITS_IN_SEGMENT[row["segment_id"]]
    if row["split"] == "validation" and "train" in earlier:
        EXCLUDED.append((i, "same_drive_as_train"))
    elif row["split"] == "test" and earlier & {"train", "validation"}:
        EXCLUDED.append((i, "same_drive_as_earlier_split"))

def input_key(i):
    return hashlib.blake2b(ARRAYS["X_sequence"][i].tobytes() + ARRAYS["X_basic"][i].tobytes(), digest_size=16).digest()

KEYS = [input_key(i) for i in range(len(ROWS))]
removed = {i for i, _ in EXCLUDED}
seen_by_split = {split: set() for split in ("train", "validation", "test")}
for i, row in enumerate(ROWS):
    if i not in removed:
        seen_by_split[row["split"]].add(KEYS[i])
kept_in_split = {split: set() for split in ("train", "validation", "test")}
for i, row in enumerate(ROWS):
    if i in removed:
        continue
    split = row["split"]
    if split == "validation" and KEYS[i] in seen_by_split["train"]:
        EXCLUDED.append((i, "input_matches_train"))
    elif split == "test" and (KEYS[i] in seen_by_split["train"] or KEYS[i] in seen_by_split["validation"]):
        EXCLUDED.append((i, "input_matches_earlier_split"))
    elif KEYS[i] in kept_in_split[split]:
        EXCLUDED.append((i, "duplicate_within_split"))
    else:
        kept_in_split[split].add(KEYS[i])
print(f"누수·중복 방지로 제외한 표본: {dict(Counter(r for _, r in EXCLUDED))}")

# %%
# 사람 판정을 모읍니다: reviews 폴더의 CSV(sample_id, reviewer, status)와 검토 Colab이 저장한 판정 파일.
VALID_STATUS = {"confirmed", "rejected", "uncertain", "exclude"}

def collect_votes(folder):
    votes = defaultdict(dict)   # sample_id → {검토자: 판정}
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") in VALID_STATUS:
                    votes[row["sample_id"]][row.get("reviewer") or path.stem] = row["status"]
    for path in sorted(folder.glob("votes/*/*.json")):
        vote = json.loads(path.read_text())
        if vote.get("status") in VALID_STATUS:
            votes[vote["sample_id"]][path.parent.name] = vote["status"]
    return votes

def decide(statuses):
    """여러 검토자의 판정을 하나로: 있음·없음이 갈리면 conflict, 보류만 있으면 uncertain."""
    s = set(statuses)
    if "exclude" in s:
        return "exclude"
    if {"confirmed", "rejected"} <= s:
        return "conflict"
    return "confirmed" if "confirmed" in s else "rejected" if "rejected" in s else "uncertain"

VOTES = collect_votes(REVIEW_DIR)
index_of = {r["sample_id"]: i for i, r in enumerate(ROWS)}
LABELS = np.array([int(r["auto_label"]) for r in ROWS], dtype=np.int8)
SOURCE = np.array(["rule"] * len(ROWS), dtype=object)
STATUS = np.array(["unreviewed"] * len(ROWS), dtype=object)
review_summary = Counter()
for sample_id, by_reviewer in VOTES.items():
    i = index_of.get(sample_id)
    if i is None:
        review_summary["not_in_dataset"] += 1
        continue
    final = decide(by_reviewer.values())
    STATUS[i] = final
    split = ROWS[i]["split"]
    review_summary[(split, final if split == "train" else "reviewed")] += 1   # validation·test는 판정 결과를 출력하지 않음
    if final in ("confirmed", "rejected"):
        LABELS[i], SOURCE[i] = (1 if final == "confirmed" else 0), "human"
    else:
        EXCLUDED.append((i, f"human_{final}"))
print("사람 판정:", dict(review_summary))
KEEP = np.ones(len(ROWS), dtype=bool)
KEEP[[i for i, _ in EXCLUDED]] = False
SPLIT = np.array([r["split"] for r in ROWS])
print({split: int((KEEP & (SPLIT == split)).sum()) for split in ("train", "validation", "test")}, "건을 데이터셋에 남깁니다.")

# %% [markdown]
# ## 8. Feature engineering (정해진 수식, 학습 없음)
#
# 시계열과 BEV에서 **99개 feature**를 계산합니다. 수식만 적용하고 데이터로부터 값을 배우지 않으므로 모든 분할에 같은 방식으로 계산해도 누수가 없습니다.
#
# - **움직임 64개:** 현재 거리·속도·상대 차량 크기, 0.2·0.5·1초 전과 비교한 변화율, 지난 1초 요약·추세, 등속 가정 1~3초 뒤 예상 간격 (각도 ±π 경계 보정, 기준 차량 크기 지름길 제거)
# - **도로·주변 차량 35개:** BEV에서 차로 중심선을 따라가 계산한 차로 기준 위치·기울기, 차선까지 거리, 앞쪽 차로 합류 여부, 앞차·끼어들 자리까지 거리
#
# 상대 차량이 기준 차량의 오른쪽에 있으면 좌우를 뒤집어, 어느 쪽에서 오든 "기준 차량 쪽으로 다가옴"이 같은 부호가 되게 합니다.

# %%
# 움직임 64개와 도로·주변 차량 35개 feature를 계산하는 함수입니다(각 feature 옆 주석 참고).
<<FEATURES>>

# %%
# 모든 표본의 99개 feature를 계산합니다. 결과를 Drive에 저장해 두고, 표본·수식이 같으면 다음 실행에서 다시 읽습니다.
started = time.monotonic()
FEATURE_CACHE = PER_FILE_DIR / "engineered_features.npz"
ROWS_HASH = hashlib.sha256(("99_unwrap_stdref\n" + "\n".join(r["sample_id"] for r in ROWS)).encode()).hexdigest()

def _map_chunk(bounds):
    a, b = bounds
    return build_map_features(ARRAYS["X_sequence"][a:b], ARRAYS["X_bev_packed"][a:b])

cached = None
if FEATURE_CACHE.is_file():
    with np.load(FEATURE_CACHE, allow_pickle=False) as data:
        if str(data["rows_hash"]) == ROWS_HASH:
            cached = {key: data[key] for key in data.files}
if cached:
    X_ENGINEERED, MOTION_NAMES, ROAD_NAMES = cached["X"], cached["motion_names"].tolist(), cached["road_names"].tolist()
else:
    motion, MOTION_NAMES = kinematic_features(ARRAYS["X_sequence"])
    if WORKERS > 1 and len(ROWS) >= 2048:
        edges = np.linspace(0, len(ROWS), WORKERS * 4 + 1, dtype=int)
        spans = [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1) if edges[i] < edges[i + 1]]
        with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as pool:
            parts = list(pool.map(_map_chunk, spans))
        road, ROAD_NAMES = np.vstack([v for v, _ in parts]), parts[0][1]
    else:
        road, ROAD_NAMES = build_map_features(ARRAYS["X_sequence"], ARRAYS["X_bev_packed"])
    X_ENGINEERED = np.column_stack((motion, road)).astype(np.float32)
    save_npz(FEATURE_CACHE, rows_hash=np.array(ROWS_HASH), X=X_ENGINEERED,
             motion_names=np.array(MOTION_NAMES), road_names=np.array(ROAD_NAMES))
ENGINEERED_NAMES = MOTION_NAMES + ROAD_NAMES
N_FEATURES = len(ENGINEERED_NAMES)
assert X_ENGINEERED.shape == (len(ROWS), N_FEATURES) and len(set(ENGINEERED_NAMES)) == N_FEATURES
print(f"feature {N_FEATURES}개 {'저장본 사용' if cached else '계산'} · {time.monotonic() - started:.0f}초")

# %% [markdown]
# ## 9. Feature 선택 (train만 사용)
#
# validation·test는 쓰지 않습니다. train 안에서 **주행 구간 단위 5-fold 교차검증**으로 판단합니다(같은 주행이 학습 fold와 검증 fold에 나뉘지 않게).
#
# 1. **결측·상수:** train에서 결측이 50%를 넘거나 값이 하나뿐인 feature 제외
# 2. **상관:** train에서 절대 상관계수 0.95 이상인 쌍은 앞쪽(기본에 가까운) feature만 남김
# 3. **중요도:** 남은 feature로 트리 부스팅을 학습하고, 검증 fold에서 순열 중요도(Average Precision 감소량)를 계산. 5개 fold 평균이 0보다 큰 feature를 선택
#
# 결과는 \`selected\` 목록으로 저장됩니다. 99개 전체도 데이터셋에 남기므로, 각 모델 담당자가 두 목록을 비교할 수 있습니다.

# %%
# train의 결측·상수 feature를 확인하고, 상관이 매우 높은 feature를 걸러냅니다.
TRAIN = KEEP & (SPLIT == "train")
X_train, y_train = X_ENGINEERED[TRAIN].astype(np.float64), LABELS[TRAIN].astype(int)
train_groups = np.array([r["segment_id"] for r in ROWS])[TRAIN]   # 교차검증 묶음 = 주행 구간
missing = np.isnan(X_train).mean(axis=0)
filled = np.where(np.isnan(X_train), np.nanmedian(X_train, axis=0), X_train)
constant = np.nanstd(filled, axis=0) == 0
step1 = [j for j in range(N_FEATURES) if missing[j] <= 0.5 and not constant[j]]
corr = np.corrcoef(filled[:, step1], rowvar=False)
step2, dropped_corr = [], {}
for a, j in enumerate(step1):
    match = next((step1[b] for b in range(a) if step1[b] in step2 and abs(corr[a, b]) >= 0.95), None)
    if match is None:
        step2.append(j)
    else:
        dropped_corr[ENGINEERED_NAMES[j]] = ENGINEERED_NAMES[match]
print(f"train {len(y_train):,}건 (양성 {y_train.sum():,}) · 결측·상수 제외 후 {len(step1)}개 · 상관 필터 후 {len(step2)}개")

# %%
# train 안에서 주행 구간 단위 5-fold 교차검증으로 순열 중요도를 계산해 feature를 고릅니다.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

def boosting():
    return HistGradientBoostingClassifier(learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=60, max_iter=200,
                                          l2_regularization=1.0, early_stopping=False, random_state=SEED)

def grouped_cv(columns, with_importance=False):
    folds = GroupKFold(n_splits=min(5, len(set(train_groups))))
    ap, importance = [], []
    for fit, held in folds.split(X_train, y_train, groups=train_groups):
        model = boosting().fit(X_train[fit][:, columns], y_train[fit])
        ap.append(average_precision_score(y_train[held], model.predict_proba(X_train[held][:, columns])[:, 1]))
        if with_importance:
            result = permutation_importance(model, X_train[held][:, columns], y_train[held], scoring="average_precision",
                                            n_repeats=3, random_state=SEED, n_jobs=-1)
            importance.append(result.importances_mean)
    return np.array(ap), (np.mean(importance, axis=0) if with_importance else None)

# 교차검증 결과는 train 표본·라벨이 같으면 Drive 저장본을 다시 씁니다(런타임이 끊겨도 처음부터 하지 않도록).
started = time.monotonic()
SELECTION_CACHE = REPORT_DIR / "selection_cache.json"
TRAIN_HASH = hashlib.sha256(json.dumps([[r["sample_id"], int(l)] for r, l, k in zip(ROWS, LABELS, TRAIN) if k]).encode()).hexdigest()
saved = json.loads(SELECTION_CACHE.read_text()) if SELECTION_CACHE.is_file() else {}
if saved.get("train_hash") == TRAIN_HASH and saved.get("step2") == step2:
    ap_corr, importance, ap_all, ap_selected = (np.array(saved[k]) for k in ("ap_corr", "importance", "ap_all", "ap_selected"))
    SELECTED = saved["selected"]
else:
    ap_corr, importance = grouped_cv(step2, with_importance=True)
    SELECTED = [j for j, value in zip(step2, importance) if value > 0] or list(step2)   # 하나도 없으면 상관 필터 목록 사용
    ap_all, _ = grouped_cv(list(range(N_FEATURES)))
    ap_selected, _ = grouped_cv(SELECTED)
    SELECTION_CACHE.write_text(json.dumps({"train_hash": TRAIN_HASH, "step2": step2, "selected": SELECTED, "importance": importance.tolist(),
                                           "ap_corr": ap_corr.tolist(), "ap_all": ap_all.tolist(), "ap_selected": ap_selected.tolist()}))
SELECTION = {
    "method": "train only; GroupKFold by drive segment; HistGradientBoosting; permutation importance (average precision)",
    "dropped_missing_or_constant": [ENGINEERED_NAMES[j] for j in range(N_FEATURES) if j not in step1],
    "dropped_by_correlation": dropped_corr,
    "importance_after_correlation_filter": {ENGINEERED_NAMES[j]: float(v) for j, v in zip(step2, importance)},
    "selected": [ENGINEERED_NAMES[j] for j in SELECTED],
    "train_cv_average_precision": {"all": ap_all.tolist(), "correlation_filtered": ap_corr.tolist(), "selected": ap_selected.tolist()},
}
for name, values in ((f"전체 {N_FEATURES}개", ap_all), (f"상관 필터 {len(step2)}개", ap_corr), (f"선택 {len(SELECTED)}개", ap_selected)):
    print(f"train 교차검증 AP · {name:12}: {values.mean():.3f} ± {values.std():.3f}")
print(f"{time.monotonic() - started:.0f}초 소요")

# %%
# train만으로 feature 선택 결과와 분포를 그림으로 남깁니다.
import matplotlib.pyplot as plt

order = np.argsort(importance)[::-1][:25]
fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True, facecolor="white")
axes[0].barh([ENGINEERED_NAMES[step2[k]] for k in order][::-1], importance[order][::-1], color="#2f6f95")
axes[0].set(title="Train CV permutation importance (top 25)", xlabel="Drop in average precision")
labels = [f"All {N_FEATURES}", f"Corr. filtered {len(step2)}", f"Selected {len(SELECTED)}"]
axes[1].boxplot([ap_all, ap_corr, ap_selected], tick_labels=labels)
axes[1].set(title="Train 5-fold CV by drive segment", ylabel="Average precision")
fig.savefig(REPORT_DIR / "feature_selection.png", dpi=150)
plt.show()

top = [ENGINEERED_NAMES[step2[k]] for k in order[:6]]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True, facecolor="white")
for ax, name in zip(axes.flat, top):
    values = X_train[:, ENGINEERED_NAMES.index(name)]
    lo, hi = np.nanpercentile(values, [1, 99])
    bins = np.linspace(lo, hi if hi > lo else lo + 1, 30)
    for label, color in ((0, "#2f6f95"), (1, "#d2652d")):
        v = values[(y_train == label) & np.isfinite(values)]
        ax.hist(np.clip(v, lo, hi), bins=bins, weights=np.full(len(v), 100 / max(len(v), 1)), histtype="step", lw=1.8,
                color=color, label=f"label {label}")
    ax.set(title=name, ylabel="Share within class (%)")
axes.flat[0].legend()
fig.suptitle("Train only: class-wise distribution of top features")
fig.savefig(REPORT_DIR / "train_top_features.png", dpi=150)
plt.show()

# %% [markdown]
# ## 10. 사람 검증 목록 만들기
#
# 각 원본 파일에서 자동 양성·자동 음성을 **무작위로 최대 4건씩** 뽑아 \`review_queue.csv\`를 만듭니다. 파일 하나가 검토 배치 하나입니다. 같은 파일의 장면을 한 번에 불러오므로 검토가 빠릅니다.
#
# - 뽑는 순서는 표본 번호의 고정 해시로 정해서, 모델 결과나 사람이 고른 사례가 섞이지 않습니다.
# - validation·test 배치는 **라벨 정확도를 확인하는 용도**입니다. 검토자는 모델 점수를 보지 않고, 검토 결과는 라벨 수정에만 씁니다.
# - 양성과 음성을 같은 수만큼 뽑았으므로, 결과를 전체 라벨 정확도로 바꿀 때는 원래 양성·음성 비율로 가중합니다.

# %%
# 파일마다 자동 양성·음성을 무작위로 최대 4건씩 골라 검토 배치를 만듭니다.
PER_CLASS = 4
QUEUE = []
by_file = defaultdict(list)
for i, row in enumerate(ROWS):
    if KEEP[i] and SOURCE[i] == "rule":
        by_file[row["source_file"]].append(i)
for source_file in sorted(by_file):
    for label in (1, 0):
        members = sorted((i for i in by_file[source_file] if int(ROWS[i]["auto_label"]) == label),
                         key=lambda i: stable_number(f"review:{ROWS[i]['sample_id']}"))
        for i in members[:PER_CLASS]:
            row = ROWS[i]
            QUEUE.append({"batch": f"{row['split']}-{source_file.split('-')[1]}", "sample_id": row["sample_id"],
                          "split": row["split"], "scenario_id": row["scenario_id"], "reference_id": row["reference_id"],
                          "target_id": row["target_id"], "anchor_frame": row["anchor_frame"],
                          "source_file": source_file, "source_generation": SOURCES[PREFIX + source_file][1]})
QUEUE.sort(key=lambda q: (q["batch"], stable_number(f"order:{q['sample_id']}")))
print(f"검토 목록 {len(QUEUE)}건 · 배치 {len({q['batch'] for q in QUEUE})}개 (분할별 {dict(Counter(q['split'] for q in QUEUE))})")

# %% [markdown]
# ## 11. 저장과 최종 검사
#
# | 파일 | 내용 |
# |---|---|
# | \`labels.csv\` | sample_id, split, label, label_source(rule/human), review_status |
# | \`samples.csv\` | 장면·차량 번호, 기준 시점, 원본 파일, **미래 정보(진입 시각 등, 입력으로 쓰지 말 것)** |
# | \`features_basic.csv\` | 기본 숫자 8개 |
# | \`inputs.npz\` | sample_id, X_basic, X_sequence, X_bev_packed, X_engineered(99개) |
# | \`feature_names.json\` | 각 입력의 이름, engineering 99개(움직임 64 + 도로 35), train 기준 선택 목록 |
# | \`feature_selection.json\` | train 교차검증 기반 선택 근거 |
# | \`split_plan.json\`, \`dataset_info.json\` | 분할 계획, 표본 수, 처리 통계, 규칙 |
# | \`excluded.csv\` | 제외한 표본과 이유 |
# | \`review_queue.csv\` | 사람 검증 배치 |

# %%
# 남길 표본만 골라 CSV·배열·설명 파일을 저장하고, 파일 사이 정합성을 다시 검사합니다.
def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

order = np.flatnonzero(KEEP)
kept = [ROWS[i] for i in order]
label_rows = [{"sample_id": ROWS[i]["sample_id"], "split": ROWS[i]["split"], "label": int(LABELS[i]),
               "label_source": SOURCE[i], "review_status": STATUS[i]} for i in order]
SAMPLE_FIELDS = ["sample_id", "split", "source_file", "scenario_id", "reference_id", "target_id", "reference_type",
                 "segment_id", "anchor_frame", "auto_label", "entry_frame", "entry_time_s", "lead_time_s", "entry_lane_id",
                 "entry_forward_center_m", "reference_lane_id", "target_lane_id", "matches_in_scene"]
write_csv(DATASET_DIR / "labels.csv", list(label_rows[0]), label_rows)
write_csv(DATASET_DIR / "samples.csv", SAMPLE_FIELDS, kept)
write_csv(DATASET_DIR / "features_basic.csv", ["sample_id", *BASIC_FEATURES],
          [{"sample_id": ROWS[i]["sample_id"], **dict(zip(BASIC_FEATURES, ARRAYS["X_basic"][i].tolist()))} for i in order])
write_csv(DATASET_DIR / "excluded.csv", ["sample_id", "split", "auto_label", "reason"],
          [{"sample_id": ROWS[i]["sample_id"], "split": ROWS[i]["split"], "auto_label": ROWS[i]["auto_label"], "reason": reason}
           for i, reason in EXCLUDED])
write_csv(DATASET_DIR / "review_queue.csv", list(QUEUE[0]), QUEUE)
save_npz(DATASET_DIR / "inputs.npz", sample_id=np.array([r["sample_id"] for r in kept]),
         X_basic=ARRAYS["X_basic"][order], X_sequence=ARRAYS["X_sequence"][order],
         X_bev_packed=ARRAYS["X_bev_packed"][order], X_engineered=X_ENGINEERED[order])
(DATASET_DIR / "feature_names.json").write_text(json.dumps({
    "X_basic": BASIC_FEATURES, "X_sequence_fields": SEQUENCE_FIELDS, "X_sequence_shape": "N x 11 frames (-1.0 s .. 0 s) x 2 vehicles (reference, target) x 7",
    "X_sequence_note": "vehicle 0 length_m/width_m is constant (5.286 x 2.332 m) for waymo_sdc; mask or ignore ref dimensions in sequence models",
    "X_bev_channels": BEV_CHANNELS, "X_bev_note": "np.unpackbits(X_bev_packed, axis=-1, count=96) -> N x 5 x 48 x 96, 1 m cells, column = x + 24, row = 24 - y",
    "X_engineered": ENGINEERED_NAMES, "X_engineered_motion": MOTION_NAMES, "X_engineered_road": ROAD_NAMES,
    "selected_engineered": SELECTION["selected"]}, indent=2, ensure_ascii=False))
(DATASET_DIR / "feature_selection.json").write_text(json.dumps(SELECTION, indent=2, ensure_ascii=False))
(DATASET_DIR / "split_plan.json").write_text(PLAN_PATH.read_text())

# 분할별 표본 수. 양성 수는 train만 기록합니다(validation·test 라벨 구성은 학습 직전까지 보지 않음).
counts = {split: {"samples": int((KEEP & (SPLIT == split)).sum()), "files": len(SPLIT_FILES[split]),
                  "human_labels": int((KEEP & (SPLIT == split) & (SOURCE == "human")).sum())}
          for split in ("train", "validation", "test")}
counts["train"]["positives"] = int((KEEP & (SPLIT == "train") & (LABELS == 1)).sum())
train_pos = counts["train"]["positives"]
INFO = {"dataset": "Waymo Open Motion Dataset v1.3.1 · cut-in within 3 s", "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "rule": RULE, "seed": SEED, "split_files": {k: [v[0], v[-1]] for k, v in SPLIT_FILES.items()}, "counts": counts,
        "processing": {k: {key: value for key, value in v.items() if k == "train" or key not in ("positive_events", "clean_negative_pairs")}
                       for k, v in COVERAGE.items()},
        "excluded": dict(Counter(r for _, r in EXCLUDED)),
        "train_class_weight_for_positive": round((counts["train"]["samples"] - train_pos) / max(train_pos, 1), 3),
        "rules_for_use": ["fit only on train", "choose settings and threshold on validation", "evaluate test once at the end",
                          "never use samples.csv future fields (entry_*, lead_time_s, auto_label) as inputs"]}
(DATASET_DIR / "dataset_info.json").write_text(json.dumps(INFO, indent=2, ensure_ascii=False))

# 파일 사이 정합성 검사
with np.load(DATASET_DIR / "inputs.npz", allow_pickle=False) as data:
    saved_ids = data["sample_id"].tolist()
    assert saved_ids == [r["sample_id"] for r in label_rows]
    assert data["X_sequence"].shape == (len(saved_ids), 11, 2, 7) and data["X_bev_packed"].shape == (len(saved_ids), 5, 48, 12)
    assert data["X_engineered"].shape == (len(saved_ids), N_FEATURES)
with (DATASET_DIR / "labels.csv").open(encoding="utf-8-sig") as handle:
    assert [r["sample_id"] for r in csv.DictReader(handle)] == saved_ids
print(json.dumps(counts, indent=1))

# %%
# 이용 안내를 붙여 데이터셋 폴더 전체를 하나의 ZIP으로 묶습니다.
README = """Waymo cut-in prediction dataset (built from Waymo Open Motion Dataset v1.3.1)

Source: Waymo Open Dataset, Waymo LLC. Waymo Dataset License Agreement for Non-Commercial Use:
https://waymo.com/open/terms/  Share only with people registered with Waymo who accepted the terms.

Rules: fit on train only; choose settings and thresholds on validation; evaluate test once at the end.
Inputs: inputs.npz (X_basic, X_sequence, X_bev_packed, X_engineered) aligned with labels.csv by sample_id.
samples.csv contains future information (entry time, lead time, auto label). Use it only for analysis or as a
multi-task target (lead_time_s), never as model input.
Splits are by source file (split_plan.json). Validation/test files were fixed before any data was read.
"""
(DATASET_DIR / "README.txt").write_text(README)
ARCHIVE = ROOT / "waymo_dataset.zip"
with NamedTemporaryFile(dir=ROOT, suffix=".tmp", delete=False) as handle:
    tmp = Path(handle.name)
with ZipFile(tmp, "w") as archive:
    for path in sorted(DATASET_DIR.iterdir()):
        archive.write(path, path.name, compress_type=ZIP_STORED if path.suffix == ".npz" else ZIP_DEFLATED)
os.replace(tmp, ARCHIVE)
print(f"ZIP: {ARCHIVE} · {ARCHIVE.stat().st_size / 1e6:.0f} MB")

# %% [markdown]
# ## 12. 내려받기와 다음 단계
#
# - ZIP을 내려받아 프로젝트의 \`data/dataset/\` 폴더에 풉니다.
# - **사람 검증(선택):** 검토 Colab(\`dataset_review.ipynb\`)에서 원본 파일 하나씩 배치로 판정하면 결과 폴더의 \`reviews/votes/\`에 바로 저장됩니다.
# - **판정 반영 방법 1 (Colab):** 이 노트북을 다시 **런타임 → 모두 실행**합니다. 이미 처리한 원본 파일은 건너뛰므로 6번은 바로 끝나고, 7번부터 판정을 반영한 새 ZIP이 만들어집니다.
# - **판정 반영 방법 2 (로컬):** 검토 Colab에서 내려받은 판정 CSV를 프로젝트의 \`data/reviews/\`에 넣고 \`scripts/apply_reviews.py\`를 실행합니다.

# %%
# 완성된 ZIP을 현재 컴퓨터로 내려받습니다.
from google.colab import files
files.download(str(ARCHIVE))

