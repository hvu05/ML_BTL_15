#!/usr/bin/env python3
"""Hash valid manifest images in chunks and report exact duplicate groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


HASH_COLUMNS = ["image_id", "relative_path", "class_name", "sha256", "hash_error"]
GROUP_COLUMNS = [
    "duplicate_group_id", "sha256", "group_size", "duplicate_files_beyond_first",
    "is_cross_class", "image_id", "relative_path", "class_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size-bytes", type=int, default=1024 * 1024)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.chunk_size_bytes < 1:
        print("--chunk-size-bytes must be positive", file=sys.stderr)
        return 2
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    required_columns = {"image_id", "relative_path", "class_name", "is_valid"}
    if missing := required_columns - set(manifest_rows[0] if manifest_rows else []):
        print(f"Manifest is missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: list[dict[str, object]] = []
    for row in manifest_rows:
        if row["is_valid"] != "True":
            continue
        record: dict[str, object] = {
            "image_id": row["image_id"], "relative_path": row["relative_path"],
            "class_name": row["class_name"], "sha256": "", "hash_error": "",
        }
        try:
            record["sha256"] = sha256_file(args.dataset_root / row["relative_path"], args.chunk_size_bytes)
        except OSError as exc:
            record["hash_error"] = f"{type(exc).__name__}: {exc}"
        hashes.append(record)

    write_csv(args.output_dir / "image_hashes.csv", HASH_COLUMNS, hashes)
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in hashes:
        if record["sha256"]:
            by_hash[str(record["sha256"])].append(record)

    group_rows: list[dict[str, object]] = []
    duplicate_group_count = 0
    duplicate_file_count = 0
    cross_class_group_count = 0
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        duplicate_group_count += 1
        duplicate_file_count += len(members) - 1
        cross_class = len({str(member["class_name"]) for member in members}) > 1
        cross_class_group_count += cross_class
        for member in members:
            group_rows.append(
                {
                    "duplicate_group_id": f"exact_{digest[:16]}", "sha256": digest,
                    "group_size": len(members), "duplicate_files_beyond_first": len(members) - 1,
                    "is_cross_class": cross_class, "image_id": member["image_id"],
                    "relative_path": member["relative_path"], "class_name": member["class_name"],
                }
            )
    write_csv(args.output_dir / "exact_duplicate_groups.csv", GROUP_COLUMNS, group_rows)

    hash_errors = sum(bool(record["hash_error"]) for record in hashes)
    print(f"Valid manifest images considered: {len(hashes)}")
    print(f"Hash errors: {hash_errors}")
    print(f"Exact duplicate groups (size >= 2): {duplicate_group_count}")
    print(f"Duplicate files beyond one representative per group: {duplicate_file_count}")
    print(f"Cross-class exact duplicate groups: {cross_class_group_count}")
    print(f"Hashes: {args.output_dir / 'image_hashes.csv'}")
    print(f"Groups: {args.output_dir / 'exact_duplicate_groups.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
