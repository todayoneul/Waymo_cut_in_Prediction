# %% [markdown]
# # Waymo 끼어들기 데이터셋 · 배치 검토
#
# 데이터셋 만들기 노트북이 고른 **검토 배치**를 하나씩 애니메이션으로 보고 판정합니다. 버튼을 누를 때마다 판정 한 건이 Google Drive에 바로 저장되고, 데이터셋을 다시 만들 때 자동으로 반영됩니다.
#
# **사용 순서**
#
# 1. 1번 준비 셀에서 \`REVIEWER_ID\`(내 이름표, 영문·숫자)와 \`MY_BATCHES\`(맡은 배치 번호 범위, 예: \`10-19\`)를 정하고 **런타임 → 모두 실행**합니다. Drive 연결과 Waymo 로그인 창이 뜨면 허용합니다.
# 2. 4번 검토 화면에서 **파란색 기준 차량(REF)** 과 **주황색 상대 차량(TARGET)** 을 봅니다.
#    - **끼어들기 있음:** 0초에 옆 차로에 있던 상대 차량이 3초 안에 기준 차량 **앞쪽** 같은 차로로 들어와 잠시 유지함
#    - **끼어들기 없음:** 3초 동안 각자 자기 차로를 유지함 (기준 차량이 옮겨 간 경우, 상대 차량이 기준 차량 뒤로 들어온 경우도 없음)
#    - **판단 보류:** 차선이 끊기거나 움직임이 애매해서 확신하기 어려움 → 이 표본은 데이터셋에서 빠짐
# 3. 배치 하나(원본 파일 하나, 최대 8건)를 끝내면 4번 마지막 셀을 다시 실행해 다음 배치를 엽니다. 언제 멈춰도 같은 이름표로 다시 열면 이어서 진행됩니다.
#
# **알아 둘 점**
#
# - 화면에는 자동 라벨이나 모델 점수를 표시하지 않습니다. 보이는 움직임만으로 판정합니다.
# - 기본 검토 대상은 \`train\`입니다. validation·test 배치는 **라벨 정확도를 확인하는 용도**이고, 모델을 학습하기 전에 한 번만 검토합니다.
# - 팀원은 공유받은 \`waymo_dataset\` 폴더를 **내 드라이브에 바로가기로 추가**한 뒤 사용합니다. Waymo 원본을 읽으려면 각자 Waymo Open Dataset 이용 등록이 된 계정으로 로그인해야 합니다.

# %% [markdown]
# ## 1. 준비

# %%
# Colab에 재생 화면과 원본 읽기에 필요한 라이브러리를 설치합니다.
%pip -q install "google-cloud-storage>=2.16,<4" "grpcio-tools==1.76.0" "protobuf>=6.31.1,<7" "google-crc32c>=1.5,<2" "pillow>=10.1" ipywidgets

# %%
# 검토자 이름표, 검토할 분할, 맡은 배치 범위를 정하고 Drive의 데이터셋 폴더에서 검토 목록을 읽습니다.
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import csv, hashlib, json, math, os, re, struct, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ipywidgets as widgets
from IPython.display import display
from google.colab import drive, output

REVIEWER_ID = ""              # @param {type:"string"}
REVIEW_SPLIT = "train"        # @param ["train", "validation", "test"]
MY_BATCHES = ""               # @param {type:"string"}
DATASET_FOLDER = "/content/drive/MyDrive/waymo_dataset"   # @param {type:"string"}

if not re.fullmatch(r"[A-Za-z0-9_-]{2,32}", REVIEWER_ID):
    raise ValueError("REVIEWER_ID에 내 이름표를 넣고 다시 실행하세요. 영문·숫자·-·_ 로 2~32자입니다. 예: gyuhan")
if MY_BATCHES.strip() and not re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", MY_BATCHES):
    raise ValueError("MY_BATCHES는 맡은 배치 번호 범위입니다. 예: 10-19 (train-00010~train-00019). 전체를 보려면 비워 둡니다.")
drive.mount("/content/drive")
output.enable_custom_widget_manager()

ROOT = Path(DATASET_FOLDER)
QUEUE_PATH = ROOT / "dataset" / "review_queue.csv"
if not QUEUE_PATH.is_file():
    raise FileNotFoundError(f"{QUEUE_PATH}가 없습니다. 데이터셋 만들기 노트북을 먼저 끝까지 실행했는지, 폴더 경로가 맞는지 확인하세요.")
with QUEUE_PATH.open(encoding="utf-8-sig", newline="") as handle:
    QUEUE = [row for row in csv.DictReader(handle) if row["split"] == REVIEW_SPLIT]
if MY_BATCHES.strip():  # 맡은 번호 범위의 배치만 남긴다
    LOW, HIGH = (int(x) for x in MY_BATCHES.split("-"))
    QUEUE = [row for row in QUEUE if LOW <= int(row["batch"].split("-")[-1]) <= HIGH]
VOTE_DIR = ROOT / "reviews" / "votes" / REVIEWER_ID
print(f"{REVIEW_SPLIT} 검토 목록 {len(QUEUE)}건 · 배치 {len({r['batch'] for r in QUEUE})}개 · 판정 저장 위치 {VOTE_DIR}")

# %%
# Waymo 이용 권한이 있는 계정으로 로그인합니다(원본 장면을 다시 읽기 위해 필요).
from google.colab import auth
from google.cloud import storage
import google.auth

BUCKET = "waymo_open_dataset_motion_v_1_3_1"
PREFIX = "uncompressed/scenario/training/"
SCHEMA_REV = "99a4cb3ff07e2fe06c2ce73da001f850f628e45a"
auth.authenticate_user()
CREDENTIALS, PROJECT = google.auth.default()
GCS = storage.Client(project=PROJECT or "", credentials=CREDENTIALS)
print("로그인 완료")

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
        payload, tail = stream.read(length), stream.read(4)
        if len(payload) != length or len(tail) != 4 or masked_crc(payload) != struct.unpack("<I", tail)[0]:
            raise ValueError("TFRecord 장면이 잘렸거나 손상되었습니다.")
        yield payload

# %% [markdown]
# ## 2. 판정 저장과 배치 고르기
#
# 판정 한 건은 \`reviews/votes/<검토자>/<표본 번호 해시>.json\` 파일 하나입니다. 검토자마다 폴더가 따로 있어서 여러 명이 동시에 검토해도 서로 덮어쓰지 않습니다. 같은 표본을 다시 판정하면 내 이전 판정만 바뀝니다.

# %%
# 판정 한 건을 Drive에 저장하고, 다시 읽어서 제대로 저장됐는지 확인합니다.
STATUS_LABEL = {"confirmed": "끼어들기 있음", "rejected": "끼어들기 없음", "uncertain": "판단 보류"}

def vote_path(sample_id):
    return VOTE_DIR / (hashlib.blake2b(sample_id.encode(), digest_size=10).hexdigest() + ".json")

def save_vote(row, status, memo=""):
    vote = {"sample_id": row["sample_id"], "status": status, "reviewer": REVIEWER_ID, "batch": row["batch"],
            "split": row["split"], "memo": memo.strip(), "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = vote_path(row["sample_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8") as handle:
        json.dump(vote, handle, ensure_ascii=False)
    os.replace(handle.name, path)
    if json.loads(path.read_text(encoding="utf-8")) != vote:
        raise IOError("저장한 판정을 다시 읽지 못했습니다.")
    return vote

def my_votes():
    votes = {}
    for path in VOTE_DIR.glob("*.json"):
        vote = json.loads(path.read_text(encoding="utf-8"))
        votes[vote["sample_id"]] = vote
    return votes

# %%
# 배치별로 내가 판정한 건수를 보여주고, 아직 끝나지 않은 첫 배치를 고릅니다.
def batch_progress():
    votes = my_votes()
    batches = defaultdict(list)
    for row in QUEUE:
        batches[row["batch"]].append(row)
    done = {name: sum(r["sample_id"] in votes for r in rows) for name, rows in batches.items()}
    return batches, done, votes

def next_batch():
    batches, done, _ = batch_progress()
    finished = sum(done[name] == len(rows) for name, rows in batches.items())
    print(f"{REVIEWER_ID} · {REVIEW_SPLIT} 배치 {finished}/{len(batches)}개 완료 · 판정 {sum(done.values())}/{len(QUEUE)}건")
    return next((name for name in sorted(batches) if done[name] < len(batches[name])), None)

next_batch();

# %% [markdown]
# ## 3. 장면 그리기
#
# 기준 시점의 기준 차량을 원점으로, 진행 방향을 오른쪽으로 놓고 그립니다. 굵은 흰색·노란색은 실제 차선, 얇은 회색 점선은 차로 중심 보조선, 보라색은 중앙 분리대, 초록색은 도로 가장자리입니다.

# %%
# 배치에 필요한 장면만 원본 파일에서 찾아 메모리에 보관합니다(원본 파일 하나를 앞에서부터 읽음).
def load_scenes(rows):
    source_file, generation = rows[0]["source_file"], int(rows[0]["source_generation"])
    needed = {r["scenario_id"] for r in rows}
    found = {}
    blob = GCS.bucket(BUCKET).blob(PREFIX + source_file, generation=generation)
    with blob.open("rb", chunk_size=32 * 1024**2, timeout=120) as stream:
        for payload in iter_records(stream):
            scene = Scenario.FromString(payload)
            if scene.scenario_id in needed:
                found[scene.scenario_id] = scene
                if len(found) == len(needed):
                    break
    missing = needed - set(found)
    if missing:
        raise LookupError(f"원본에서 장면을 찾지 못했습니다: {sorted(missing)[:3]}")
    return found

# %%
# 지도 선을 종류별(차로 중심, 차선, 중앙 분리대, 도로 가장자리) 색과 모양으로 나눕니다.
MAP_STYLE = {
    "center": ("#7c90a3", "dot", 1), "edge": ("#67c4aa", "solid", 3), "median": ("#dda3f3", "solid", 5),
    "unknown": ("#9cb1c1", "solid", 2), "white_solid": ("#ffffff", "solid", 4), "white_dash": ("#ffffff", "dash", 4),
    "yellow_solid": ("#ffcf64", "solid", 4), "yellow_dash": ("#ffcf64", "dash", 4),
}

def to_local(origin):
    c, s = math.cos(origin.heading), math.sin(origin.heading)
    rot, center = np.array([[c, -s], [s, c]]), [origin.center_x, origin.center_y]
    return lambda xy: (np.asarray(xy) - center) @ rot

def line_kind(kind, obj):
    if kind == "lane":
        return "center"
    t = int(obj.type)
    if kind == "road_edge":
        return "median" if t == 2 else "edge"
    if t in (1, 2, 3):
        return "white_dash" if t == 1 else "white_solid"
    if t in (4, 5, 6, 7, 8):
        return "yellow_dash" if t in (4, 5) else "yellow_solid"
    return "unknown"

def road_lines(scene, transform, bounds, road_z):
    bands = {key: ([], []) for key in MAP_STYLE}
    xl, xr, yl, yr = bounds
    for feature in scene.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind not in ("lane", "road_line", "road_edge"):
            continue
        obj = getattr(feature, kind)
        coords = np.array([[p.x, p.y, p.z] for p in obj.polyline])
        if len(coords) < 2:
            continue
        xy = transform(coords[:, :2])
        low, high = np.minimum(xy[:-1], xy[1:]), np.maximum(xy[:-1], xy[1:])
        same_level = (np.abs(coords[:-1, 2] - road_z) < 3.5) & (np.abs(coords[1:, 2] - road_z) < 3.5)   # 고가도로 등 다른 높이 제외
        keep = (low[:, 0] < xr + 3) & (high[:, 0] > xl - 3) & (low[:, 1] < yr + 3) & (high[:, 1] > yl - 3) & same_level
        xs, ys = bands[line_kind(kind, obj)]
        ix = np.flatnonzero(keep)
        for run in np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1):
            if len(run):
                part = xy[run[0]:run[-1] + 2]
                xs.extend([*part[:, 0], None])
                ys.extend([*part[:, 1], None])
    return [(kind, *xy, *MAP_STYLE[kind]) for kind, xy in bands.items() if xy[0]]

# %%
# 차량 한 대의 외곽선(앞쪽 뾰족)과, 두 차량이 모두 보이는 화면 범위를 계산합니다.
def vehicle_outline(track, index, transform):
    if index >= len(track.states) or not track.states[index].valid:
        return None
    s = track.states[index]
    forward = np.array([math.cos(s.heading), math.sin(s.heading)])
    left = np.array([-forward[1], forward[0]])
    p = np.array([s.center_x, s.center_y])
    a, b = max(s.length, 1) / 2, max(s.width, .5) / 2
    return transform([p + a * forward + b * left, p + (a + .45) * forward, p + a * forward - b * left,
                      p - a * forward - b * left, p - a * forward + b * left, p + a * forward + b * left])

def view_bounds(tracks, indices, transform):
    points = [s for track in tracks for j in indices if j < len(track.states) and (s := track.states[j]).valid]
    xy = transform([[s.center_x, s.center_y] for s in points])
    xl, yl = xy.min(axis=0) - [8, 8]
    xr, yr = xy.max(axis=0) + [11, 8]
    mid, half = (yl + yr) / 2, max((yr - yl) / 2, 9)
    return float(xl), float(xr), float(mid - half), float(mid + half)

# %%
# 도로 배경과 범례를 한 번 그려 두고, 점선·실선을 구분해 그리는 함수입니다.
def draw_line(draw, points, color, style, size):
    width = max(1, round(size))
    if style == "solid":
        draw.line(points, fill=color, width=width)
        return
    on, off = (2., 7.) if style == "dot" else (15., 10.)
    phase = 0.
    for p, q in zip(points, points[1:]):
        length = math.dist(p, q)
        if length <= 1e-6:
            continue
        cursor = -phase
        while cursor < length:
            start, end = max(0., cursor), min(length, cursor + on)
            if end > start:
                at = lambda a: (p[0] + (q[0] - p[0]) * a / length, p[1] + (q[1] - p[1]) * a / length)
                draw.line([at(start), at(end)], fill=color, width=width)
            cursor += on + off
        phase = (phase + length) % (on + off)

def background(bounds, road, show_centers=True):
    width, height = 1040, 405
    xl, xr, yl, yr = bounds
    scale = min((width - 72) / (xr - xl), (height - 100) / (yr - yl))
    pixel = lambda p: (width / 2 + (p[0] - (xl + xr) / 2) * scale, (height + 20) / 2 - (p[1] - (yl + yr) / 2) * scale)
    image = Image.new("RGB", (width, height), "#253546")
    draw = ImageDraw.Draw(image)
    for kind, xs, ys, color, style, size in road:
        if kind == "center" and not show_centers:
            continue
        points = []
        for a, b in [*zip(xs, ys), (None, None)]:
            if a is None:
                if len(points) >= 2:
                    draw_line(draw, points, color, style, size)
                points = []
            else:
                points.append(pixel((a, b)))
    font = ImageFont.load_default(size=15)
    draw.rectangle((20, 33, width - 20, 77), fill="#1b2b3b")
    legend = [(35, "REFERENCE", "#67a5ff", "solid", 5), (176, "TARGET", "#ff9a58", "solid", 5),
              (292, "MARKED LINES", "#ffffff", "dash", 4), (460, "LANE CENTER (GUIDE)", "#a5b8c8", "dot", 1),
              (680, "MEDIAN", "#dda3f3", "solid", 5), (807, "ROAD EDGE", "#67c4aa", "solid", 3)]
    for x, label, color, style, size in legend:
        draw_line(draw, [(x, 53), (x + 31, 53)], color, style, size)
        draw.text((x + 38, 45), label, font=font, fill="#eef4f8")
    ruler = 10 * scale
    draw.line([(35, height - 25), (35 + ruler, height - 25)], fill="#dae5ee", width=2)
    draw.text((42 + ruler, height - 35), "10 m", font=font, fill="#dae5ee")
    return image, pixel

# %%
# 표본 하나의 -1초~+3.2초 프레임을 그리는 함수를 만듭니다(자동 라벨은 표시하지 않음).
def scene_frames(scene, row):
    cars = {str(track.id): track for track in scene.tracks}
    ref, target = cars[str(row["reference_id"])], cars[str(row["target_id"])]
    t0 = int(row["anchor_frame"])
    origin = ref.states[t0]
    times = np.asarray(scene.timestamps_seconds) - scene.timestamps_seconds[t0]
    frames = [int(j) for j in np.flatnonzero((times >= -1.005) & (times <= 3.205))]
    transform = to_local(origin)
    bounds = view_bounds([ref, target], frames, transform)
    road = road_lines(scene, transform, bounds, origin.center_z)
    cache, title_font, car_font = {}, ImageFont.load_default(size=18), ImageFont.load_default(size=12)

    def paint(j, show_centers=True):
        if show_centers not in cache:
            cache[show_centers] = background(bounds, road, show_centers)
        base, pixel = cache[show_centers]
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((18, 2, image.width - 18, 32), fill="#253546")
        draw.text((34, 5), f"Time {times[j]:+.1f}s", fill="#f4f7fb", font=title_font)
        for track, label, color in ((ref, "REF", "#67a5ff"), (target, "TARGET", "#ff9a58")):
            corners = vehicle_outline(track, j, transform)
            if corners is not None:
                xy = [pixel(p) for p in corners]
                draw.polygon(xy, fill=color, outline="#ffffff", width=2)
                draw.text((xy[1][0] + 5, xy[1][1] - 16), label, font=car_font, fill="#f3f7fb", stroke_width=2, stroke_fill="#253546")
        return image
    return [(j, float(times[j])) for j in frames], t0, paint

def to_png(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()

def to_gif(images):
    stream = BytesIO()
    images[0].save(stream, format="GIF", save_all=True, append_images=images[1:], duration=110, disposal=2)
    return stream.getvalue()

# %%
# 재생 버튼, 시간 슬라이더, 중심선 표시 여부로 현재 장면 이미지를 바꾸는 재생기입니다.
def make_player():
    picture = widgets.Image(format="png", layout=widgets.Layout(width="1040px", max_width="100%"))
    seek = widgets.SelectionSlider(description="시간", options=[("0.0초", 0)], continuous_update=False,
                                   layout=widgets.Layout(width="420px", max_width="100%"))
    centers = widgets.Checkbox(value=True, description="차로 중심선 표시", indent=False)
    play, zero = widgets.Button(description="▶ 재생"), widgets.Button(description="0초로")
    state = {"scene": None, "busy": False, "gif": None}

    def draw(j):
        if state["scene"] is not None and not state["busy"]:
            picture.format, picture.value = "png", to_png(state["scene"][2](j, centers.value))

    def play_now(_):
        if state["scene"] is None:
            return
        frames, t0, paint = state["scene"]
        if state["gif"] is None:
            state["gif"] = to_gif([paint(j, centers.value) for j, _ in frames if j >= t0])
        picture.value = b""
        picture.format, picture.value = "gif", state["gif"]

    def load(value):
        state.update(scene=value, busy=True, gif=None)
        for control in (seek, play, zero, centers):
            control.disabled = value is None
        try:
            if value is None:
                picture.value = b""
                return
            frames, t0, _ = value
            seek.options = [(f"{t:+.1f}초", j) for j, t in frames]
            seek.value = t0
        finally:
            state["busy"] = False
        draw(t0)

    def jump_to_zero(_):
        seek.value = state["scene"][1]
        draw(seek.value)

    def toggle_centers(_):
        state["gif"] = None
        draw(seek.value)

    seek.observe(lambda change: draw(change["new"]), names="value")
    centers.observe(toggle_centers, names="value")
    play.on_click(play_now)
    zero.on_click(jump_to_zero)
    load(None)
    return widgets.VBox([widgets.HBox([play, zero, centers]), seek, picture]), load

# %% [markdown]
# ## 4. 검토 화면
#
# 아래 셀을 실행하면 아직 끝나지 않은 첫 배치가 열립니다. 배치를 끝낸 뒤 **이 셀을 다시 실행**하면 다음 배치가 열립니다. 특정 배치를 다시 보고 싶으면 \`BATCH\`에 배치 이름(예: \`train-00012\`)을 적습니다.

# %%
# 배치 하나를 열어 장면을 보여주고, 버튼을 누르면 판정을 저장한 뒤 다음 장면으로 넘어갑니다.
def show_batch(name):
    batches, _, votes = batch_progress()
    rows = batches[name]
    print(f"배치 {name}: 원본 {rows[0]['source_file']}에서 장면 {len({r['scenario_id'] for r in rows})}개를 찾는 중… (1~2분 걸릴 수 있음)")
    scenes = load_scenes(rows)
    player, load = make_player()
    header, message = widgets.HTML(), widgets.Label()
    memo = widgets.Text(description="메모", placeholder="선택: 애매했던 이유", layout=widgets.Layout(width="520px"))
    buttons = {status: widgets.Button(description=label, button_style=style) for status, label, style in
               [("confirmed", "끼어들기 있음", "success"), ("rejected", "끼어들기 없음", "danger"), ("uncertain", "판단 보류", "warning")]}
    previous, following = widgets.Button(description="◀ 이전"), widgets.Button(description="다음 ▶")
    state = {"pos": next((i for i, r in enumerate(rows) if r["sample_id"] not in votes), 0)}

    def render(note=""):
        row = rows[state["pos"]]
        mine = votes.get(row["sample_id"], {})
        done = sum(r["sample_id"] in votes for r in rows)
        header.value = (f"<b>{name}</b> · {state['pos'] + 1}/{len(rows)}번째 · 이 배치 판정 {done}/{len(rows)} · "
                        f"내 판정: <b>{STATUS_LABEL.get(mine.get('status'), '아직 없음')}</b>")
        memo.value = mine.get("memo", "")
        previous.disabled, following.disabled = state["pos"] == 0, state["pos"] == len(rows) - 1
        load(scene_frames(scenes[row["scenario_id"]], row))
        message.value = note

    def mark(status):
        row = rows[state["pos"]]
        try:
            votes[row["sample_id"]] = save_vote(row, status, memo.value)
        except Exception as error:
            message.value = f"저장하지 못했습니다. 다시 눌러 주세요: {error}"
            return
        left = [i for i, r in enumerate(rows) if r["sample_id"] not in votes]
        if left:
            state["pos"] = left[0]
            render(f"저장했습니다: {STATUS_LABEL[status]}")
        else:
            render(f"저장했습니다. 배치 {name} 완료! 이 셀을 다시 실행하면 다음 배치가 열립니다.")

    def move(step):
        state["pos"] = min(max(state["pos"] + step, 0), len(rows) - 1)
        render()

    for status, button in buttons.items():
        button.on_click(lambda _, s=status: mark(s))
    previous.on_click(lambda _: move(-1))
    following.on_click(lambda _: move(1))
    display(widgets.VBox([header, player, memo, widgets.HBox(list(buttons.values())), widgets.HBox([previous, following]), message]))
    render()

# %%
# 다음 배치를 엽니다. 특정 배치를 다시 보려면 BATCH에 이름을 적습니다.
BATCH = ""   # @param {type:"string"}
name = BATCH.strip() or next_batch()
if name is None:
    print("이 분할의 검토 배치를 모두 끝냈습니다.")
elif name not in {r["batch"] for r in QUEUE}:
    print(f"{name} 배치가 {REVIEW_SPLIT} 검토 목록에 없습니다.")
else:
    show_batch(name)

# %% [markdown]
# ## 5. 판정 반영
#
# 판정은 이미 Drive에 저장되어 있으므로 따로 할 일은 없습니다. 새 데이터셋이 필요할 때 아래 둘 중 하나를 합니다.
#
# - **Colab:** 데이터셋 만들기 노트북(\`dataset_build.ipynb\`)을 다시 **모두 실행**합니다. 처리한 원본은 건너뛰고 판정을 반영한 ZIP을 새로 만듭니다.
# - **로컬:** 아래 셀로 모든 검토자의 판정을 CSV로 내려받아 프로젝트의 \`data/reviews/\`에 넣고 \`scripts/apply_reviews.py\`를 실행합니다.

# %%
# 모든 검토자의 판정을 CSV 한 개로 모아 내려받습니다. 내려받으려면 EXPORT_CSV를 체크한 뒤 실행하세요.
from google.colab import files

EXPORT_CSV = False   # @param {type:"boolean"}

rows = []
for path in sorted((ROOT / "reviews" / "votes").glob("*/*.json")):
    vote = json.loads(path.read_text(encoding="utf-8"))
    rows.append({key: vote.get(key, "") for key in ("sample_id", "reviewer", "status", "split", "batch", "memo", "saved_utc")})
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
export = Path(f"/content/reviews_{stamp}.csv")
with export.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["sample_id", "reviewer", "status", "split", "batch", "memo", "saved_utc"])
    writer.writeheader()
    writer.writerows(rows)
print(f"판정 {len(rows)}건 · 검토자 {len({r['reviewer'] for r in rows})}명 · {dict(Counter(r['status'] for r in rows))}")
if EXPORT_CSV and rows:
    files.download(str(export))
elif not EXPORT_CSV:
    print("내려받기를 건너뛰었습니다. CSV를 받으려면 EXPORT_CSV를 체크하고 이 셀만 실행하세요.")

