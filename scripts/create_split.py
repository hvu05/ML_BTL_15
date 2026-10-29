#!/usr/bin/env python3
"""Create a deterministic group-aware, class-stratified 4/1/1 split from clean manifest."""
from __future__ import annotations
import argparse, csv, random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest", type=Path, default=Path("outputs/b1/clean_manifest.csv")); p.add_argument("--output", type=Path, default=Path("dataset/split.csv")); p.add_argument("--summary", type=Path, default=Path("dataset/split_summary.csv")); p.add_argument("--seed", type=int, default=42); a = p.parse_args()
    with a.manifest.open(encoding="utf-8", newline="") as f: rows = [r for r in csv.DictReader(f) if r["is_valid"] == "True" and r["is_in_clean_dataset"] == "True"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        near = r.get("near_duplicate_group_id", ""); exact = r.get("duplicate_group_id", "")
        r["split_group_id"], r["split_group_source"] = (near, "near_duplicate") if near else ((exact, "exact_duplicate") if exact else (r["image_id"], "image_id"))
        groups[r["split_group_id"]].append(r)
    for gid, members in groups.items():
        if len({r["class_name"] for r in members}) != 1: raise SystemExit(f"Conflicting labels in group {gid}; resolve manually before splitting")
    ratios = {"train": 4 / 6, "val": 1 / 6, "test": 1 / 6}; output_rows = []
    by_label: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for members in groups.values(): by_label[members[0]["class_name"]].append(members)
    for label, label_groups in sorted(by_label.items()):
        rng = random.Random(f"{a.seed}:{label}"); rng.shuffle(label_groups); label_groups.sort(key=len, reverse=True)
        targets = {name: len([r for group in label_groups for r in group]) * ratio for name, ratio in ratios.items()}; counts = {name: 0 for name in ratios}
        for members in label_groups:
            split = min(ratios, key=lambda name: ((counts[name] + len(members) - targets[name]) ** 2, counts[name], name))
            counts[split] += len(members)
            for r in members: output_rows.append({"image_id": r["image_id"], "relative_path": r["relative_path"], "class_name": label, "split": split, "split_group_id": r["split_group_id"], "split_group_source": r["split_group_source"], "sha256": r["sha256"]})
    output_rows.sort(key=lambda r: r["image_id"]); a.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "relative_path", "class_name", "split", "split_group_id", "split_group_source", "sha256"]
    with a.output.open("w", encoding="utf-8", newline="") as f: w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(output_rows)
    summary = defaultdict(int)
    for r in output_rows: summary[(r["class_name"], r["split"])] += 1
    with a.summary.open("w", encoding="utf-8", newline="") as f: w = csv.DictWriter(f, fieldnames=["class_name", "split", "image_count"]); w.writeheader(); w.writerows({"class_name": c, "split": s, "image_count": n} for (c, s), n in sorted(summary.items()))
    print(f"Rows split: {len(output_rows)}; groups: {len(groups)}; output: {a.output}"); return 0


if __name__ == "__main__": raise SystemExit(main())
