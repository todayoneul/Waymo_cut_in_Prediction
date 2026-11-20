"""Waymo Motion 원본(TFRecord 안의 Scenario)을 읽고 차량 좌표를 다루는 함수."""
import struct

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def read_tfrecord(path):
    # TFRecord: [길이 8바이트][CRC 4바이트][데이터][CRC 4바이트]가 반복된다.
    with open(path, "rb") as handle:
        while header := handle.read(12):
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            handle.read(4)
            yield payload


def load_scene(raw_dir, scene_id):
    pool = descriptor_pool.DescriptorPool()
    for item in descriptor_pb2.FileDescriptorSet.FromString((raw_dir / "scenario.desc").read_bytes()).file:
        pool.Add(item)
    scenario = message_factory.GetMessageClass(pool.FindMessageTypeByName("waymo.open_dataset.Scenario"))
    for path in sorted(raw_dir.glob("*.tfrecord")):
        for payload in read_tfrecord(path):
            scene = scenario.FromString(payload)
            if scene.scenario_id == scene_id:
                return scene, path.name
    raise LookupError(f"원본 장면을 찾지 못했습니다: {scene_id}")


def track_states(track):
    # 각 시점: x, y, 방향(rad), vx, vy, 길이, 너비, 유효 여부
    return np.array([[s.center_x, s.center_y, s.heading, s.velocity_x, s.velocity_y,
                      s.length, s.width, s.valid] for s in track.states], dtype=float)


def map_items(scene):
    items = []
    for feature in scene.map_features:
        kind = feature.WhichOneof("feature_data")
        obj = getattr(feature, kind)
        if kind in ("lane", "road_line", "road_edge"):
            points = [[p.x, p.y] for p in obj.polyline]
        elif kind in ("crosswalk", "speed_bump", "driveway"):
            points = [[p.x, p.y] for p in obj.polygon]
        elif kind == "stop_sign":
            points = [[obj.position.x, obj.position.y]]
        else:
            continue
        if points:
            items.append((kind, int(getattr(obj, "type", 0)), np.array(points, dtype=float)))
    return items


def frame_of(state):
    """기준 차량의 한 시점을 원점, 진행 방향을 +x로 하는 좌표 변환."""
    x, y, heading = state[0], state[1], state[2]
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, -s], [s, c]])
    return lambda xy: (np.asarray(xy, dtype=float) - [x, y]) @ rotation, heading


def vehicle_outline(state, transform, heading0):
    x, y, heading, _, _, length, width, _ = state
    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-forward[1], forward[0]])
    center = np.array([x, y])
    corners = [center + length / 2 * forward + width / 2 * left, center + (length / 2 + .6) * forward,
               center + length / 2 * forward - width / 2 * left, center - length / 2 * forward - width / 2 * left,
               center - length / 2 * forward + width / 2 * left]
    return transform(corners)
