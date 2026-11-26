# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "numpy"]
# ///
"""Verify a BEV shard against the shared dataset and animate one pair per class (train only)."""
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bev_sequence import OFFSETS, RESOLUTIONS, read_frames, shard_rows


def animate(arrays, destination, sid, label):
    palette = [(86, 103, 117), (188, 197, 202), (64, 163, 255), (255, 155, 58), (109, 124, 138)]
    levels = {key: read_frames(arrays, key) for key in RESOLUTIONS}
    frames = []
    for i, offset in enumerate(OFFSETS):
        page = Image.new("RGB", (960, 286), (22, 29, 35))
        for col, (name, sequence) in enumerate(levels.items()):
            mask = sequence[i]
            rgb = np.full((*mask.shape[1:], 3), (22, 29, 35), dtype=np.uint8)
            for channel in (0, 1, 4, 2, 3):
                rgb[mask[channel] != 0] = palette[channel]
            page.paste(Image.fromarray(rgb).resize((480, 240), Image.Resampling.NEAREST), (col * 480, 23))
            ImageDraw.Draw(page).text((col * 480 + 8, 5), f"{name}    {offset / 10:+.1f} s", fill="white")
        ImageDraw.Draw(page).text((8, 268), f"{sid}   label={label}   Blue: reference   Orange: target", fill="white")
        frames.append(page)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=[180] * 10 + [700], loop=0)


def check(path, dataset):
    with np.load(dataset / "inputs.npz", allow_pickle=False) as shared:
        indices = {str(sid): i for i, sid in enumerate(shared["sample_id"])}
        sequence = shared["X_sequence"]
    with ZipFile(path) as archive:
        meta = json.loads(archive.read("manifest.json"))
        records = meta["samples"]
        expected = {r["sample_id"]: r for r in shard_rows(dataset, meta["source"])}
        ids = {r["sample_id"] for r in records}
        if len(ids) != len(records) or not ids or not ids <= expected.keys():
            raise ValueError("Invalid sample IDs")
        if not meta["excerpt"] and ids != expected.keys():
            raise ValueError("Incomplete shard")
        np.testing.assert_allclose(meta["offsets_s"], OFFSETS / 10)
        split = records[0]["split"]
        counts = {"0": 0, "1": 0}
        for rec in records:
            sid = rec["sample_id"]
            row = expected[sid]
            for field in ("label", "label_source", "split", "segment_id", "time_target_s"):
                if rec[field] != row[field]:
                    raise ValueError(f"{sid}: {field}")
            with np.load(io.BytesIO(archive.read(rec["member"])), allow_pickle=False) as data:
                xy = sequence[indices[sid], :, :, :2]
                np.testing.assert_allclose(data["centers"], xy, atol=1e-4, err_msg=sid)
                for key, step in RESOLUTIONS.items():
                    h, w = round(48 / step), round(96 / step)
                    masks = data["cars_" + key]
                    if masks.shape != (11, 3, h, w // 8) or data["map_" + key].shape != (2, h, w // 8):
                        raise ValueError(f"{sid}: {key} shape")
                    ij = np.rint((xy * [1, -1] + [24, 24]) / step).astype(int)
                    x, y = ij[..., 0], ij[..., 1]
                    t, c = np.where((x >= 0) & (x < w) & (y >= 0) & (y < h))
                    if not ((masks[t, c, y[t, c], x[t, c] // 8] >> (7 - x[t, c] % 8)) & 1).all():
                        raise ValueError(f"{sid}: misplaced vehicle at {key}")
                label = rec["label"]
                if split == "train" and not counts[label]:
                    animate(data, path.with_name(path.stem + f"_label{label}.gif"), sid, label)
                counts[label] += 1
    if split == "train":
        print(f"{path.name}: {len(records)} IDs verified (train), negative={counts['0']}, positive={counts['1']}, excerpt={meta['excerpt']}")
    else:
        print(f"{path.name}: {len(records)} IDs verified ({split}), excerpt={meta['excerpt']}")


if __name__ == "__main__":
    check(Path(sys.argv[1]), ROOT / "data/dataset")
