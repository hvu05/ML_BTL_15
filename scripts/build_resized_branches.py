#!/usr/bin/env python3
"""Build 224×224 direct and neutral-gray padded branches from clean_manifest.csv."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

GRAY_BGR = (128, 128, 128)
EXTRA = ["source_image_id", "source_relative_path", "resized_relative_path", "branch", "target_width", "target_height", "resized_content_width", "resized_content_height", "resize_scale", "interpolation", "padding_mode", "padding_bgr", "pad_top", "pad_bottom", "pad_left", "pad_right", "is_resize_success", "resize_error_message"]
FAIL_COLUMNS = ["source_image_id", "source_relative_path", "class_name", "branch", "resize_error_message"]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("outputs/b1/clean_manifest.csv"))
    p.add_argument("--dataset-root", type=Path, default=Path("Sohas_weapon-Classification"))
    p.add_argument("--output-root", type=Path, default=Path("dataset"))
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--clean-output", action="store_true", help="Clear only generated branches, features, and stale temporary buffers before building.")
    return p.parse_args()


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)


def safe_path(value: str) -> Path:
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(value)


def interpolation(width: int, height: int, target: int) -> tuple[int, str]:
    return (cv2.INTER_CUBIC, "INTER_CUBIC") if width < target and height < target else (cv2.INTER_AREA, "INTER_AREA")


def direct(image: Any, target: int) -> tuple[Any, dict[str, Any]]:
    h, w = image.shape[:2]; method, name = interpolation(w, h, target)
    return cv2.resize(image, (target, target), interpolation=method), {"resized_content_width": target, "resized_content_height": target, "resize_scale": "", "interpolation": name, "padding_mode": "none", "padding_bgr": "", "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}


def padded(image: Any, target: int) -> tuple[Any, dict[str, Any]]:
    h, w = image.shape[:2]; scale = min(target / w, target / h)
    nw, nh = max(1, min(target, round(w * scale))), max(1, min(target, round(h * scale)))
    method, name = interpolation(w, h, target)
    resized = image if (nw, nh) == (w, h) else cv2.resize(image, (nw, nh), interpolation=method)
    left, right = (target - nw) // 2, target - nw - (target - nw) // 2
    top, bottom = (target - nh) // 2, target - nh - (target - nh) // 2
    output = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=GRAY_BGR)
    return output, {"resized_content_width": nw, "resized_content_height": nh, "resize_scale": round(scale, 10), "interpolation": name, "padding_mode": "neutral_gray_constant", "padding_bgr": "128,128,128", "pad_top": top, "pad_bottom": bottom, "pad_left": left, "pad_right": right}


def ensure_original_link(output: Path, source: Path) -> None:
    link = output / "original"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace existing original entry: {link}")
    else:
        link.symlink_to(os.path.relpath(source.resolve(), output.resolve()), target_is_directory=True)


def main() -> int:
    a = args(); output, source = a.output_root.resolve(), a.dataset_root.resolve()
    with a.manifest.open(encoding="utf-8", newline="") as f: all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["is_valid"] == "True" and r["is_in_clean_dataset"] == "True"]
    output.mkdir(parents=True, exist_ok=True); ensure_original_link(output, source)
    if a.clean_output:
        for p in (output / "resized" / "direct", output / "resized" / "padding", output / "features"):
            if p.exists(): shutil.rmtree(p)
        for p in output.glob("sohas_hog_*"):
            if p.is_dir(): shutil.rmtree(p)
    roots = {"direct": output / "resized" / "direct", "padding": output / "resized" / "padding"}
    for root in roots.values(): (root / "images").mkdir(parents=True, exist_ok=True)
    records, failures = {k: [] for k in roots}, {k: [] for k in roots}
    for i, row in enumerate(rows, 1):
        relative = safe_path(row["relative_path"])
        try:
            image = cv2.imread(str(source / relative), cv2.IMREAD_COLOR)
            if image is None: raise ValueError("cv2.imread returned None")
        except (OSError, ValueError, cv2.error) as exc:
            message = f"{type(exc).__name__}: {exc}"
            for branch in roots: failures[branch].append({"source_image_id": row["image_id"], "source_relative_path": row["relative_path"], "class_name": row["class_name"], "branch": branch, "resize_error_message": message})
            continue
        for branch, transform in (("direct", direct), ("padding", padded)):
            rec: dict[str, Any] = dict(row); rec.update({key: "" for key in EXTRA}); rec.update({"source_image_id": row["image_id"], "source_relative_path": row["relative_path"], "branch": branch, "target_width": a.target_size, "target_height": a.target_size, "is_resize_success": False})
            try:
                result, meta = transform(image, a.target_size)
                if result.shape[:2] != (a.target_size, a.target_size): raise RuntimeError(f"Unexpected shape {result.shape[:2]}")
                out_rel = (Path("images") / relative).as_posix(); out_path = roots[branch] / out_rel; out_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out_path), result): raise OSError("cv2.imwrite returned False")
                rec.update(meta); rec.update({"resized_relative_path": out_rel, "is_resize_success": True})
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                message = f"{type(exc).__name__}: {exc}"; rec["resize_error_message"] = message
                failures[branch].append({"source_image_id": row["image_id"], "source_relative_path": row["relative_path"], "class_name": row["class_name"], "branch": branch, "resize_error_message": message})
            records[branch].append(rec)
        if i % 500 == 0 or i == len(rows): print(f"Processed {i}/{len(rows)} images", flush=True)
    for branch, root in roots.items():
        write_csv(root / "resized_manifest.csv", list(all_rows[0]) + EXTRA, records[branch]); write_csv(root / "resize_failures.csv", FAIL_COLUMNS, failures[branch])
        print(f"{branch}: successes={sum(r['is_resize_success'] is True for r in records[branch])}, failures={len(failures[branch])}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
