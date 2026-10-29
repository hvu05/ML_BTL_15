#!/usr/bin/env python3
"""Apply transparent B1 cleaning rules without modifying raw dataset files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


CLEAN_COLUMNS = [
    "image_id", "relative_path", "class_name", "width", "height", "aspect_ratio", "format", "mode",
    "file_size_bytes", "is_image_extension", "is_valid", "error_message", "sha256",
    "duplicate_group_id", "duplicate_group_size", "duplicate_representative_image_id",
    "duplicate_label_status", "is_in_clean_dataset", "cleaning_status", "cleaning_reason",
]
SUMMARY_COLUMNS = [
    "class_name", "candidate_files", "valid_images_before_cleaning", "invalid_or_unusable_files",
    "exact_duplicate_same_label_excluded", "cross_label_conflict_excluded", "images_in_clean_dataset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    manifest = read_csv(args.manifest)
    hashes = read_csv(args.hashes)
    hashes_by_id = {row["image_id"]: row for row in hashes}

    # Groups are derived from all successfully hashed valid images, not only a pre-filtered report.
    members_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        hash_row = hashes_by_id.get(row["image_id"])
        if row["is_valid"] == "True" and hash_row and hash_row["sha256"] and not hash_row["hash_error"]:
            members_by_hash[hash_row["sha256"]].append(row)

    duplicate_info: dict[str, dict[str, str | int | bool]] = {}
    for digest, members in members_by_hash.items():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda row: row["relative_path"])
        labels = {row["class_name"] for row in members}
        group_id = f"exact_{digest[:16]}"
        cross_label = len(labels) > 1
        for member in members:
            duplicate_info[member["image_id"]] = {
                "group_id": group_id,
                "size": len(members),
                "representative": ordered[0]["image_id"],
                "cross_label": cross_label,
            }

    clean_rows: list[dict[str, str | int | bool]] = []
    for row in manifest:
        hash_row = hashes_by_id.get(row["image_id"], {})
        record: dict[str, str | int | bool] = {column: row.get(column, "") for column in CLEAN_COLUMNS}
        record["sha256"] = hash_row.get("sha256", "")
        record.update({
            "duplicate_group_id": "", "duplicate_group_size": "", "duplicate_representative_image_id": "",
            "duplicate_label_status": "not_duplicate", "is_in_clean_dataset": False,
            "cleaning_status": "excluded", "cleaning_reason": "",
        })
        if row["is_valid"] != "True":
            record["cleaning_reason"] = "invalid_or_unreadable_image"
        elif not row["class_name"]:
            record["cleaning_reason"] = "missing_class_label"
        elif hash_row.get("hash_error"):
            record["cleaning_reason"] = "hashing_failed"
        else:
            info = duplicate_info.get(row["image_id"])
            if not info:
                record.update({"is_in_clean_dataset": True, "cleaning_status": "included", "cleaning_reason": "unique_content"})
            else:
                record.update({
                    "duplicate_group_id": info["group_id"], "duplicate_group_size": info["size"],
                    "duplicate_representative_image_id": info["representative"],
                })
                if info["cross_label"]:
                    record.update({
                        "duplicate_label_status": "cross_label_conflict", "cleaning_status": "needs_manual_review",
                        "cleaning_reason": "exact_duplicate_with_conflicting_labels",
                    })
                elif row["image_id"] == info["representative"]:
                    record.update({
                        "duplicate_label_status": "same_label_duplicate", "is_in_clean_dataset": True,
                        "cleaning_status": "included", "cleaning_reason": "representative_of_exact_duplicate_group",
                    })
                else:
                    record.update({
                        "duplicate_label_status": "same_label_duplicate", "cleaning_status": "excluded",
                        "cleaning_reason": "exact_duplicate_same_label_non_representative",
                    })
        clean_rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "clean_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_COLUMNS)
        writer.writeheader()
        writer.writerows(clean_rows)

    class_names = sorted({str(row["class_name"]) for row in clean_rows if row["class_name"]})
    summary_rows: list[dict[str, int | str]] = []
    for class_name in class_names + ["__overall__"]:
        rows = clean_rows if class_name == "__overall__" else [row for row in clean_rows if row["class_name"] == class_name]
        summary_rows.append({
            "class_name": class_name,
            "candidate_files": len(rows),
            "valid_images_before_cleaning": sum(row["is_valid"] == "True" for row in rows),
            "invalid_or_unusable_files": sum(row["cleaning_reason"] in {"invalid_or_unreadable_image", "missing_class_label", "hashing_failed"} for row in rows),
            "exact_duplicate_same_label_excluded": sum(row["cleaning_reason"] == "exact_duplicate_same_label_non_representative" for row in rows),
            "cross_label_conflict_excluded": sum(row["cleaning_reason"] == "exact_duplicate_with_conflicting_labels" for row in rows),
            "images_in_clean_dataset": sum(row["is_in_clean_dataset"] is True for row in rows),
        })
    with (args.output_dir / "cleaning_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    included = sum(row["is_in_clean_dataset"] is True for row in clean_rows)
    same_label_excluded = sum(row["cleaning_reason"] == "exact_duplicate_same_label_non_representative" for row in clean_rows)
    conflicts = sum(row["cleaning_reason"] == "exact_duplicate_with_conflicting_labels" for row in clean_rows)
    print(f"Manifest rows processed: {len(clean_rows)}")
    print(f"Images in clean dataset: {included}")
    print(f"Same-label exact duplicates excluded: {same_label_excluded}")
    print(f"Cross-label duplicate records needing review: {conflicts}")
    print(f"Clean manifest: {args.output_dir / 'clean_manifest.csv'}")
    print(f"Cleaning summary: {args.output_dir / 'cleaning_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
