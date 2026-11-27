"""Verify BEV sequence manifests against the local reviewed table and split plan."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="Waymo_cut_in_Prediction repository root")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--bev", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--crc", action="store_true", help="Read and CRC-check every outer ZIP member")
    a = p.parse_args()
    repo = a.repo.resolve()
    if a.dataset is None:
        a.dataset = repo / "data/dataset"
    if a.bev is None:
        a.bev = repo / "data/bev_sequences"
    plan = json.loads((a.dataset / "split_plan.json").read_text(encoding="utf-8"))
    membership = {Path(k).name: v["split"] for k, v in plan["files"].items()}
    with (a.dataset / "labels.csv").open(encoding="utf-8-sig") as f:
        labels = {}
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            if sid in labels or row["label"] not in ("0", "1"):
                raise ValueError(f"Duplicate ID or invalid label: {sid}")
            labels[sid] = row
    expected = defaultdict(dict)
    groups = {}
    with (a.dataset / "samples.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid, source, sp = row["sample_id"], row["source_file"], row["split"]
            label = labels.pop(sid)
            if sp != label["split"] or sp != membership[source] or groups.setdefault(row["segment_id"], sp) != sp:
                raise ValueError(f"Split mismatch: {sid}")
            lead = None
            if label["label"] == row["auto_label"] == "1" and row["lead_time_s"]:
                lead = float(row["lead_time_s"])
            expected[source][sid] = (sp, row["segment_id"], label["label"], label["label_source"], lead)
    if labels or set(expected) != membership.keys():
        raise ValueError("Label, sample and split-plan coverage differ")
    result = []
    for source, sp in sorted(membership.items()):
        number = source.split("-")[-3]
        path = a.bev / f"{sp}_{number}.zip"
        with ZipFile(path) as archive:
            raw = archive.read("manifest.json")
            manifest = json.loads(raw)
            if manifest["source"] != source or manifest["excerpt"]:
                raise ValueError(f"Source mismatch: {path.name}")
            records = manifest["samples"]
            names = {r["member"] for r in records}
            if (len(names) != len(records) or len(archive.namelist()) != len(records)+1 or
                    names | {"manifest.json"} != set(archive.namelist())):
                raise ValueError(f"Invalid archive member list: {path.name}")
            for rec in records:
                fields = tuple(rec[k] for k in ("split", "segment_id", "label", "label_source", "time_target_s"))
                if fields != expected[source].pop(rec["sample_id"]):
                    raise ValueError(f"Label or metadata mismatch: {rec['sample_id']}")
            if expected[source] or (a.crc and archive.testzip() is not None):
                raise ValueError(f"Missing samples or CRC failure: {path.name}")
            result.append(dict(file=path.name, split=sp, samples=len(records), bytes=path.stat().st_size,
                               manifest_sha256=hashlib.sha256(raw).hexdigest()))
    totals = Counter()
    for shard in result:
        totals[shard["split"]] += shard["samples"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(dict(counts=totals, crc_checked=a.crc, shards=result), f, indent=2)
    print(f"Verified {len(result)} shards: {dict(totals)}")


if __name__ == "__main__":
    main()

