#!/usr/bin/env python3
"""Extract HOG only after split.csv exists; preserve split metadata in a branch .npz."""
from __future__ import annotations
import argparse, csv, json, shutil, tempfile
from pathlib import Path
import cv2, numpy as np
from skimage.feature import hog

CONFIG = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2), "block_norm": "L2-Hys", "grayscale_conversion": "cv2.COLOR_BGR2GRAY"}
PARAMS = {k: v for k, v in CONFIG.items() if k != "grayscale_conversion"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--branch", choices=("direct", "padding"), required=True, help="Chọn nhánh HOG cần trích xuất."); p.add_argument("--split", type=Path, default=Path("dataset/split.csv")); p.add_argument("--dataset-root", type=Path, default=Path("dataset")); a = p.parse_args()
    a.resized_root = a.dataset_root / "resized" / a.branch
    a.resized_manifest = a.resized_root / "resized_manifest.csv"
    a.output = a.dataset_root / "features" / f"hog_{a.branch}.npz"
    with a.split.open(encoding="utf-8", newline="") as f: splits = {r["image_id"]: r for r in csv.DictReader(f)}
    with a.resized_manifest.open(encoding="utf-8", newline="") as f: rows = [r for r in csv.DictReader(f) if r["is_resize_success"] == "True" and r["image_id"] in splits]
    classes = sorted({r["class_name"] for r in rows}); labels = {c: i for i, c in enumerate(classes)}
    probe = hog(np.zeros((224, 224), dtype=np.uint8), **PARAMS); a.output.parent.mkdir(parents=True, exist_ok=True); temp = Path(tempfile.mkdtemp(prefix="sohas_hog_", dir=a.output.parent)); array = np.memmap(temp / "X.dat", mode="w+", dtype=np.float32, shape=(len(rows), len(probe)))
    metadata = {k: [] for k in ("y", "split", "image_id", "relative_path", "sha256", "duplicate_group_id", "duplicate_representative_image_id")}
    try:
        for i, r in enumerate(rows):
            image = cv2.imread(str(a.resized_root / r["resized_relative_path"]), cv2.IMREAD_COLOR)
            if image is None: raise RuntimeError(f"Cannot read resized image: {r['resized_relative_path']}")
            array[i] = hog(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), **PARAMS).astype(np.float32)
            s = splits[r["image_id"]]; metadata["y"].append(labels[r["class_name"]]); metadata["split"].append(s["split"])
            for key in ("image_id", "relative_path", "sha256", "duplicate_group_id", "duplicate_representative_image_id"): metadata[key].append(r.get(key, ""))
            if (i + 1) % 500 == 0 or i + 1 == len(rows): print(f"HOG {i + 1}/{len(rows)}", flush=True)
        array.flush()
        np.savez(a.output, X=array, y=np.asarray(metadata["y"], dtype=np.int64), split=np.asarray(metadata["split"]), class_names=np.asarray(classes), image_ids=np.asarray(metadata["image_id"]), relative_paths=np.asarray(metadata["relative_path"]), sha256=np.asarray(metadata["sha256"]), duplicate_group_id=np.asarray(metadata["duplicate_group_id"]), duplicate_representative_image_id=np.asarray(metadata["duplicate_representative_image_id"]), hog_config_json=np.asarray(json.dumps(CONFIG, sort_keys=True)))
    finally:
        del array; shutil.rmtree(temp, ignore_errors=True)
    print(f"Saved {len(rows)} vectors × {len(probe)} features to {a.output}"); return 0


if __name__ == "__main__": raise SystemExit(main())
