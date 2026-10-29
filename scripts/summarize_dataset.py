#!/usr/bin/env python3
"""Summarize valid images in a B1 image manifest by class."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COLUMNS = [
    "class_name",
    "valid_image_count",
    "percentage_of_valid_images",
    "width_min",
    "width_median",
    "width_max",
    "height_min",
    "height_median",
    "height_max",
    "aspect_ratio_min",
    "aspect_ratio_median",
    "aspect_ratio_max",
    "format_distribution",
    "invalid_file_count",
    "missing_label_file_count",
    "class_imbalance_ratio_max_to_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to image_manifest.csv")
    parser.add_argument("--output", type=Path, required=True, help="Path to dataset_statistics.csv")
    return parser.parse_args()


def numeric_summary(values: list[float]) -> tuple[float, float, float]:
    return min(values), statistics.median(values), max(values)


def rounded(value: float) -> float:
    return round(value, 6)


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"Manifest does not exist: {args.manifest}", file=sys.stderr)
        return 2

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    required = {"class_name", "width", "height", "aspect_ratio", "format", "is_valid"}
    missing_columns = required - set(manifest_rows[0] if manifest_rows else [])
    if missing_columns:
        print(f"Manifest is missing columns: {sorted(missing_columns)}", file=sys.stderr)
        return 2

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_by_class: Counter[str] = Counter()
    missing_label_count = 0
    for row in manifest_rows:
        label = row["class_name"]
        if not label:
            missing_label_count += 1
        if row["is_valid"] == "True" and label:
            grouped[label].append(row)
        elif row["is_valid"] != "True":
            invalid_by_class[label or "[missing label]"] += 1

    total_valid = sum(len(rows) for rows in grouped.values())
    counts = [len(rows) for rows in grouped.values()]
    imbalance_ratio = max(counts) / min(counts) if counts and min(counts) else ""
    output_rows: list[dict[str, Any]] = []

    for class_name in sorted(grouped):
        rows = grouped[class_name]
        widths = [float(row["width"]) for row in rows]
        heights = [float(row["height"]) for row in rows]
        ratios = [float(row["aspect_ratio"]) for row in rows]
        width_min, width_median, width_max = numeric_summary(widths)
        height_min, height_median, height_max = numeric_summary(heights)
        ratio_min, ratio_median, ratio_max = numeric_summary(ratios)
        formats = dict(sorted(Counter(row["format"] for row in rows).items()))
        output_rows.append(
            {
                "class_name": class_name,
                "valid_image_count": len(rows),
                "percentage_of_valid_images": rounded(100 * len(rows) / total_valid),
                "width_min": int(width_min), "width_median": rounded(width_median), "width_max": int(width_max),
                "height_min": int(height_min), "height_median": rounded(height_median), "height_max": int(height_max),
                "aspect_ratio_min": rounded(ratio_min),
                "aspect_ratio_median": rounded(ratio_median),
                "aspect_ratio_max": rounded(ratio_max),
                "format_distribution": json.dumps(formats, sort_keys=True),
                "invalid_file_count": invalid_by_class[class_name],
                "missing_label_file_count": 0,
                "class_imbalance_ratio_max_to_min": "",
            }
        )

    if total_valid:
        all_valid = [row for rows in grouped.values() for row in rows]
        widths = [float(row["width"]) for row in all_valid]
        heights = [float(row["height"]) for row in all_valid]
        ratios = [float(row["aspect_ratio"]) for row in all_valid]
        width_min, width_median, width_max = numeric_summary(widths)
        height_min, height_median, height_max = numeric_summary(heights)
        ratio_min, ratio_median, ratio_max = numeric_summary(ratios)
        output_rows.append(
            {
                "class_name": "__overall__",
                "valid_image_count": total_valid,
                "percentage_of_valid_images": 100.0,
                "width_min": int(width_min), "width_median": rounded(width_median), "width_max": int(width_max),
                "height_min": int(height_min), "height_median": rounded(height_median), "height_max": int(height_max),
                "aspect_ratio_min": rounded(ratio_min),
                "aspect_ratio_median": rounded(ratio_median),
                "aspect_ratio_max": rounded(ratio_max),
                "format_distribution": json.dumps(dict(sorted(Counter(row["format"] for row in all_valid).items())), sort_keys=True),
                "invalid_file_count": sum(invalid_by_class.values()),
                "missing_label_file_count": missing_label_count,
                "class_imbalance_ratio_max_to_min": rounded(float(imbalance_ratio)) if imbalance_ratio else "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Valid images used for statistics: {total_valid}")
    print(f"Classes with valid images: {len(grouped)}")
    print(f"Max/min class-count ratio: {imbalance_ratio:.6f}" if imbalance_ratio else "Max/min class-count ratio: n/a")
    print(f"Statistics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
