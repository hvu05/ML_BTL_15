#!/usr/bin/env python3
"""Run inference on a single image using trained SVM pipeline."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np
from scipy.special import softmax
from skimage.feature import hog

CLASSES = ["billete", "knife", "monedero", "pistol", "smartphone", "tarjeta"]

CLASS_INFO = {
    "knife": {"vi": "Dao / Vũ khí lạnh", "type": "VŨ KHÍ NGUY HIỂM", "is_weapon": True},
    "pistol": {"vi": "Súng ngắn / Vũ khí nóng", "type": "VŨ KHÍ NGUY HIỂM", "is_weapon": True},
    "smartphone": {"vi": "Điện thoại thông minh", "type": "Đồ vật thông thường", "is_weapon": False},
    "tarjeta": {"vi": "Thẻ ngân hàng / Thẻ từ", "type": "Đồ vật thông thường", "is_weapon": False},
    "billete": {"vi": "Tiền giấy / Tiền mặt", "type": "Đồ vật thông thường", "is_weapon": False},
    "monedero": {"vi": "Ví tiền / Bóp cầm tay", "type": "Đồ vật thông thường", "is_weapon": False},
}

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def predict_image(image_path: Path, model_path: Path) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 1. Resize direct 224x224
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (w > 224 or h > 224) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (224, 224), interpolation=interp)

    # 2. HOG extraction (26,244 dims)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    feat = hog(gray, **HOG_CONFIG).astype(np.float32).reshape(1, -1)

    # 3. Model predict (Pipeline: Scaler -> PCA 256 -> SVC RBF)
    pipeline = joblib.load(model_path)
    pred_idx = int(pipeline.predict(feat)[0])
    pred_class = CLASSES[pred_idx]

    # Convert decision margins to probability distribution via Softmax
    scores = pipeline.decision_function(feat)[0]
    probs = softmax(scores)

    results = []
    for i, cname in enumerate(CLASSES):
        results.append({
            "class": cname,
            "name_vi": CLASS_INFO[cname]["vi"],
            "category": CLASS_INFO[cname]["type"],
            "is_weapon": CLASS_INFO[cname]["is_weapon"],
            "score": float(scores[i]),
            "prob": float(probs[i]),
        })

    results.sort(key=lambda x: x["prob"], reverse=True)
    return {
        "predicted_class": pred_class,
        "class_info": CLASS_INFO[pred_class],
        "top_prediction": results[0],
        "all_probabilities": results,
    }


def render_progress_bar(prob: float, width: int = 20) -> str:
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict weapon/object class for an image using SVM")
    parser.add_argument("image", type=Path, help="Path to input image file")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/svm/rbf_svc_direct_model.joblib"),
        help="Path to trained joblib pipeline",
    )
    args = parser.parse_args()

    res = predict_image(args.image, args.model)
    top = res["top_prediction"]

    print("\n" + "=" * 60)
    print(" KẾT QUẢ PHÂN LOẠI ẢNH (HOG + PCA + SVM RBF)")
    print("=" * 60)
    print(f"Ảnh đầu vào      : {args.image}")
    print(f"Dự đoán vật thể  : {top['class'].upper()} ({top['name_vi']})")
    print(f"Nhóm phân loại   : {top['category'].upper()}")
    print(f"Độ tin cậy (Prob): {top['prob'] * 100:.2f}%")
    print("-" * 60)
    print("BẢNG PHÂN BỐ XÁC SUẤT 6 LỚP:")
    for r in res["all_probabilities"]:
        bar = render_progress_bar(r["prob"], width=16)
        tag = "[VŨ KHÍ]" if r["is_weapon"] else "[THƯỜNG]"
        print(f"  {r['class']:<11} ({r['name_vi']:<22}) {tag:<9} | {bar} {r['prob']*100:>5.1f}%")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
