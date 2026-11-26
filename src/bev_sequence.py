"""Rasterize reviewed Waymo pairs in the reference vehicle's fixed anchor frame."""
import csv
import io
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

RESOLUTIONS = {"1m": 1.0, "20cm": 0.2}
OFFSETS = np.arange(-10, 1)
CHANNELS = ["lane_center", "road_lines_and_edges", "reference", "target", "other_vehicles"]
VALID_SPLITS = ("train", "validation", "test")


def _attach_targets(rows):
    for row in rows:
        if row.get("label") not in ("0", "1"):
            raise ValueError(row["sample_id"])
        has_time = row["label"] == "1" and row["auto_label"] == "1" and row["lead_time_s"]
        row["time_target_s"] = float(row["lead_time_s"]) if has_time else None
        if has_time and not 0 < row["time_target_s"] <= 3.05:
            raise ValueError(f"Invalid lead time: {row['sample_id']}")
    return rows


def rows_by_source(dataset, splits=VALID_SPLITS):
    """Load finalized samples and labels once, grouped by source TFRecord filename."""
    allowed = set(splits)
    grouped = defaultdict(list)
    by_id = {}
    with (dataset / "samples.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in allowed:
                continue
            sid = row["sample_id"]
            if sid in by_id:
                raise ValueError(f"Duplicate sample: {sid}")
            by_id[sid] = row
            grouped[row["source_file"]].append(row)
    with (dataset / "labels.csv").open(encoding="utf-8-sig") as handle:
        for label in csv.DictReader(handle):
            row = by_id.get(label["sample_id"])
            if row is None:
                continue
            if label["split"] != row["split"] or "label" in row:
                raise ValueError(label["sample_id"])
            row.update(label)
    for rows in grouped.values():
        _attach_targets(rows)
    return dict(grouped)


def shard_rows(dataset, source, split=None):
    """Join finalized labels for one source file across train, validation, or test."""
    allowed = (split,) if split is not None else VALID_SPLITS
    with (dataset / "samples.csv").open(encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["source_file"] == source and r["split"] in allowed]
    by_id = {r["sample_id"]: r for r in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError(f"Missing or duplicate samples: {source}")
    with (dataset / "labels.csv").open(encoding="utf-8-sig") as handle:
        for label in csv.DictReader(handle):
            row = by_id.get(label["sample_id"])
            if row is None:
                continue
            if label["split"] != row["split"] or "label" in row:
                raise ValueError(label["sample_id"])
            row.update(label)
    return _attach_targets(rows)


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def scene_geometry(scene):
    lines = []
    for feature in scene.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind in ("lane", "road_line", "road_edge"):
            points = np.asarray([(p.x, p.y, p.z) for p in getattr(feature, kind).polyline])
            if len(points) >= 2:
                bounds = (
                    float(points[:, 0].min()), float(points[:, 0].max()),
                    float(points[:, 1].min()), float(points[:, 1].max()),
                    float(points[:, 2].min()), float(points[:, 2].max()),
                )
                lines.append((int(kind != "lane"), points, bounds))
    n_frames = len(scene.timestamps_seconds)
    vehicles = {}
    for track in scene.tracks:
        if track.object_type != 1:
            continue
        states = np.zeros((n_frames, 6), dtype=np.float64)
        valid = np.zeros(n_frames, dtype=bool)
        for i, s in enumerate(track.states[:n_frames]):
            if s.valid:
                states[i] = (s.center_x, s.center_y, s.center_z, s.heading, s.length, s.width)
                valid[i] = True
        vehicles[track.id] = (states, valid)
    return lines, vehicles


def pair_geometry(lines, vehicles, timestamps, row):
    anchor = int(row["anchor_frame"])
    ref, target = int(row["reference_id"]), int(row["target_id"])
    if anchor < 10 or anchor >= len(timestamps):
        raise ValueError(row["sample_id"])
    times = np.asarray(timestamps[anchor - 10:anchor + 1])
    if not np.allclose(times - times[-1], OFFSETS / 10, atol=0.02):
        raise ValueError(f"Irregular timestamps: {row['sample_id']}")
    ref_states, ref_valid = vehicles[ref]
    if not ref_valid[anchor]:
        raise ValueError(f"Invalid reference state: {row['sample_id']}")
    rx, ry, rz, r_heading = ref_states[anchor, :4]
    origin, axes = np.array([rx, ry]), rotation(r_heading)

    local_lines = []
    for channel, points, (xmin, xmax, ymin, ymax, zmin, zmax) in lines:
        if zmax < rz - 3 or zmin > rz + 3 or xmax < rx - 80 or xmin > rx + 80 or ymax < ry - 80 or ymin > ry + 80:
            continue
        near = np.abs(points[:, 2] - rz) <= 3
        edges = np.flatnonzero(np.diff(np.r_[False, near, False]))
        for start, stop in edges.reshape(-1, 2):
            if stop - start < 2:
                continue
            xy = (points[start:stop, :2] - origin) @ axes
            if np.all(xy.max(0) >= [-25, -25]) and np.all(xy.min(0) <= [73, 25]):
                local_lines.append((channel, xy))

    frames = []
    centers = np.empty((11, 2, 2), dtype=np.float32)
    unit_box = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for frame, offset in enumerate(OFFSETS):
        t = anchor + int(offset)
        polygons = []
        for vid, (v_states, v_valid) in vehicles.items():
            channel = 0 if vid == ref else 1 if vid == target else 2
            if not v_valid[t]:
                if channel < 2:
                    raise ValueError(f"Invalid pair history: {row['sample_id']}")
                continue
            sx, sy, sz, s_heading, s_len, s_wid = v_states[t]
            if channel == 2 and (abs(sz - rz) > 3 or abs(sx - rx) > 85 or abs(sy - ry) > 85):
                continue
            xy = (np.array([sx, sy]) - origin) @ axes
            if channel < 2:
                centers[frame, channel] = xy
            length, width = (4.7, 2.0) if vid == ref else (max(1.0, s_len), max(0.5, s_wid))
            corners = (unit_box * [length / 2, width / 2]) @ rotation(r_heading - s_heading) + xy
            if np.all(corners.max(0) >= [-25, -25]) and np.all(corners.min(0) <= [73, 25]):
                polygons.append((channel, corners))
        frames.append(polygons)
    return local_lines, frames, centers


def render(lines, frames, resolution):
    size = (round(96 / resolution), round(48 / resolution))

    def pixels(xy):
        ij = np.rint((xy * [1, -1] + [24, 24]) / resolution).astype(int)
        return list(map(tuple, ij))

    def layers(count):
        return [Image.new("1", size) for _ in range(count)]

    def pack(images):
        return np.packbits(np.stack([np.asarray(im) for im in images]), axis=-1)

    maps = layers(2)
    for channel, points in lines:
        ImageDraw.Draw(maps[channel]).line(pixels(points), fill=1, width=1)
    moving = []
    for polygons in frames:
        cars = layers(3)
        for channel, corners in polygons:
            ImageDraw.Draw(cars[channel]).polygon(pixels(corners), fill=1)
        moving.append(pack(cars))
    return pack(maps), np.stack(moving)


def build_archive(scenes, rows, destination, source, *, excerpt=False):
    """Build on local temp storage and publish atomically once every expected ID is written."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".partial", delete=False) as tmp:
        local_partial = Path(tmp.name)
    remote_partial = None
    manifest = dict(source=source, excerpt=excerpt, offsets_s=(OFFSETS / 10).tolist(),
                    channels=CHANNELS, resolutions=RESOLUTIONS, reference_size_m=[4.7, 2.0], samples=[])
    try:
        with ZipFile(local_partial, "w", allowZip64=True) as archive:
            for scene in scenes:
                group = grouped.pop(scene.scenario_id, [])
                if not group:
                    continue
                lines, vehicles = scene_geometry(scene)
                for row in group:
                    local, frames, centers = pair_geometry(lines, vehicles, scene.timestamps_seconds, row)
                    arrays = {"centers": centers}
                    for name, resolution in RESOLUTIONS.items():
                        arrays["map_" + name], arrays["cars_" + name] = render(local, frames, resolution)
                    key = f"{len(manifest['samples']):06d}.npz"
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **arrays)
                    archive.writestr(key, buffer.getvalue())
                    fields = ("sample_id", "segment_id", "split", "label", "label_source", "time_target_s")
                    manifest["samples"].append(dict(member=key, **{k: row[k] for k in fields}))
                if not grouped:
                    break
            if grouped or not manifest["samples"]:
                raise ValueError(f"Missing scenes in {source}: {len(grouped)}")
            archive.writestr("manifest.json", json.dumps(manifest))
        with NamedTemporaryFile(dir=destination.parent, suffix=".partial", delete=False) as dst_tmp:
            remote_partial = Path(dst_tmp.name)
        shutil.copyfile(local_partial, remote_partial)
        remote_partial.replace(destination)
        remote_partial = None
    finally:
        local_partial.unlink(missing_ok=True)
        if remote_partial is not None:
            remote_partial.unlink(missing_ok=True)
    return len(manifest["samples"])


def read_frames(arrays, resolution):
    """Return (time, channel, height, width), with the static map shared across time."""
    maps = np.unpackbits(arrays["map_" + resolution], axis=-1)
    cars = np.unpackbits(arrays["cars_" + resolution], axis=-1)
    return np.concatenate([np.broadcast_to(maps, (len(cars), *maps.shape)), cars], axis=1)
