"""Copy the shareable records of a finished run into results/bev/<name> so they can be committed.

Settings, histories, metrics and figures are copied. Sample-level files (predictions.csv, groups.csv,
samples.txt) and checkpoints (*.pt) stay under data/, which Git ignores.
"""
import argparse
import shutil
from pathlib import Path

EXCLUDED = {"predictions.csv", "groups.csv"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="Finished folder under data/training_runs")
    p.add_argument("--name", help="Folder name under results/bev (default: the run folder name)")
    p.add_argument("--dest", type=Path, default=Path.cwd() / "results/bev")
    args = p.parse_args()
    files = sorted(f for f in args.run.rglob("*")
                   if f.is_file() and f.suffix in {".json", ".csv", ".png"} and f.name not in EXCLUDED)
    if not files:
        p.error(f"No records found in {args.run}")
    target = args.dest / (args.name or args.run.name)
    target.mkdir(parents=True, exist_ok=False)
    for file in files:
        copy = target / file.relative_to(args.run)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, copy)
    print(f"Copied {len(files)} files to {target}")


if __name__ == "__main__":
    main()
