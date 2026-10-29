#!/usr/bin/env python3
"""Build a reproducible, non-destructive manifest for an image dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jfif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}
MANIFEST_COLUMNS = [
    "image_id",
    "relative_path",
    "class_name",
    "width",
    "height",
    "aspect_ratio",
    "format",
    "mode",
    "file_size_bytes",
    "is_image_extension",
    "is_valid",
    "error_message",
]
ERROR_COLUMNS = ["image_id", "relative_path", "class_name", "error_category", "error_message"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root containing class folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated CSV files.")
    return parser.parse_args()


def class_from_path(relative_path: Path) -> str:
    """Use the first directory under dataset root as the class label."""
    return relative_path.parts[0] if len(relative_path.parts) >= 2 else ""


def inspect_file(dataset_root: Path, path: Path) -> tuple[dict[str, Any], str | None]:
    """Verify then fully decode one file; return its manifest row and an error category."""
    relative_path = path.relative_to(dataset_root)
    row: dict[str, Any] = {
        "image_id": relative_path.as_posix(),
        "relative_path": relative_path.as_posix(),
        "class_name": class_from_path(relative_path),
        "width": "",
        "height": "",
        "aspect_ratio": "",
        "format": "",
        "mode": "",
        "file_size_bytes": path.stat().st_size,
        "is_image_extension": path.suffix.lower() in IMAGE_EXTENSIONS,
        "is_valid": False,
        "error_message": "",
    }

    try:
        # verify() checks structural integrity but invalidates this image object.
        with Image.open(path) as image:
            image.verify()
        # Reopen and load() so truncated/decode errors are also captured.
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            row.update(
                {
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 8) if height else "",
                    "format": image.format or "",
                    "mode": image.mode or "",
                    "is_valid": True,
                }
            )
        return row, None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        category = "corrupted_or_unreadable_image" if row["is_image_extension"] else "non_image_file"
        return row, category


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"Dataset root does not exist or is not a directory: {dataset_root}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((path for path in dataset_root.rglob("*") if path.is_file()), key=lambda p: p.as_posix())

    manifest_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for path in files:
        try:
            row, error_category = inspect_file(dataset_root, path)
        except OSError as exc:
            relative_path = path.relative_to(dataset_root)
            row = {
                "image_id": relative_path.as_posix(), "relative_path": relative_path.as_posix(),
                "class_name": class_from_path(relative_path), "width": "", "height": "",
                "aspect_ratio": "", "format": "", "mode": "", "file_size_bytes": "",
                "is_image_extension": path.suffix.lower() in IMAGE_EXTENSIONS, "is_valid": False,
                "error_message": f"{type(exc).__name__}: {exc}",
            }
            error_category = "file_access_error"
        manifest_rows.append(row)
        if error_category:
            error_rows.append({
                "image_id": row["image_id"], "relative_path": row["relative_path"],
                "class_name": row["class_name"], "error_category": error_category,
                "error_message": row["error_message"],
            })

    write_csv(args.output_dir / "image_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    write_csv(args.output_dir / "corrupted_images.csv", ERROR_COLUMNS, error_rows)

    valid_count = sum(row["is_valid"] is True for row in manifest_rows)
    invalid_count = len(manifest_rows) - valid_count
    non_image_count = sum(error["error_category"] == "non_image_file" for error in error_rows)
    missing_label_count = sum(not row["class_name"] for row in manifest_rows)
    print(f"Candidate files: {len(manifest_rows)}")
    print(f"Valid decoded images: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print(f"Non-image files: {non_image_count}")
    print(f"Missing labels: {missing_label_count}")
    print(f"Manifest: {args.output_dir / 'image_manifest.csv'}")
    print(f"Errors: {args.output_dir / 'corrupted_images.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
