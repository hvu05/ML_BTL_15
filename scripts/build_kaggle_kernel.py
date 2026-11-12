import base64
import gzip
from pathlib import Path

raw_csv = Path("outputs/data_cleaning/clean_manifest.csv").read_bytes()
b64_str = base64.b64encode(gzip.compress(raw_csv)).decode("ascii")

parts = []

parts.append('''#!/usr/bin/env python3
"""End-to-end reproducible SOHAS preprocessing and SVM benchmarking pipeline for Kaggle."""

from __future__ import annotations

import base64
import csv
import gzip
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

GRAY_BGR = (128, 128, 128)
HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

CLEAN_MANIFEST_B64 = """''' + b64_str + '''"""

def unpack_clean_manifest(target_path: Path) -> Path:
    if not target_path.exists():
        print("Unpacking embedded clean_manifest.csv...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = gzip.decompress(base64.b64decode(CLEAN_MANIFEST_B64))
        target_path.write_bytes(raw_bytes)
        print("Unpacked bytes:", len(raw_bytes))
    return target_path


def find_raw_dataset_root() -> Path:
    expected_classes = {"billete", "knife", "monedero", "pistol", "smartphone", "tarjeta"}
    for root, dirs, _ in os.walk("/kaggle/input"):
        if expected_classes.issubset(set(dirs)):
            print("Found SOHAS raw classes in:", root)
            return Path(root)
    raise RuntimeError("Could not find SOHAS classes in /kaggle/input")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def resize_direct(img: np.ndarray, target_size: int = 224) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (w > target_size or h > target_size) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (target_size, target_size), interpolation=interp)
    info = {
        "target_width": target_size, "target_height": target_size,
        "resized_content_width": target_size, "resized_content_height": target_size,
        "resize_scale": round(target_size / max(w, h), 8),
        "interpolation": "cv2.INTER_AREA" if (w > target_size or h > target_size) else "cv2.INTER_LINEAR",
        "padding_mode": "none", "padding_bgr": "",
        "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0,
    }
    return resized, info


def resize_padding(img: np.ndarray, target_size: int = 224) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_h = target_size - nh
    pad_w = target_size - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    canvas = np.full((target_size, target_size, 3), GRAY_BGR, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = scaled
    info = {
        "target_width": target_size, "target_height": target_size,
        "resized_content_width": nw, "resized_content_height": nh,
        "resize_scale": round(scale, 8),
        "interpolation": "cv2.INTER_AREA" if scale < 1.0 else "cv2.INTER_LINEAR",
        "padding_mode": "neutral_gray", "padding_bgr": "128,128,128",
        "pad_top": top, "pad_bottom": bottom, "pad_left": left, "pad_right": right,
    }
    return canvas, info


def build_resized_branches(raw_root: Path, manifest_path: Path, output_root: Path) -> None:
    manifest = read_csv(manifest_path)
    clean_items = [r for r in manifest if r["is_valid"] == "True" and r["is_in_clean_dataset"] == "True"]
    print("Resizing clean images count:", len(clean_items))

    direct_root = output_root / "resized" / "direct"
    padding_root = output_root / "resized" / "padding"
    direct_rows = []
    padding_rows = []

    for i, row in enumerate(clean_items):
        img_p = raw_root / Path(row["relative_path"])
        img = cv2.imread(str(img_p))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_p}")

        d_img, d_info = resize_direct(img, 224)
        d_out = direct_root / row["relative_path"]
        d_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d_out), d_img)
        d_meta = dict(row)
        d_meta.update(d_info)
        d_meta.update({
            "source_image_id": row["image_id"],
            "source_relative_path": row["relative_path"],
            "resized_relative_path": row["relative_path"],
            "branch": "direct",
            "is_resize_success": True,
            "resize_error_message": "",
        })
        direct_rows.append(d_meta)

        p_img, p_info = resize_padding(img, 224)
        p_out = padding_root / row["relative_path"]
        p_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p_out), p_img)
        p_meta = dict(row)
        p_meta.update(p_info)
        p_meta.update({
            "source_image_id": row["image_id"],
            "source_relative_path": row["relative_path"],
            "resized_relative_path": row["relative_path"],
            "branch": "padding",
            "is_resize_success": True,
            "resize_error_message": "",
        })
        padding_rows.append(p_meta)

        if (i + 1) % 2000 == 0 or (i + 1) == len(clean_items):
            print(f"Resized {i + 1}/{len(clean_items)} images", flush=True)

    extra_cols = ["source_image_id", "source_relative_path", "resized_relative_path", "branch", "target_width", "target_height", "resized_content_width", "resized_content_height", "resize_scale", "interpolation", "padding_mode", "padding_bgr", "pad_top", "pad_bottom", "pad_left", "pad_right", "is_resize_success", "resize_error_message"]
    cols = list(manifest[0].keys()) + extra_cols
    write_csv(direct_root / "resized_manifest.csv", cols, direct_rows)
    write_csv(padding_root / "resized_manifest.csv", cols, padding_rows)
    print("Done resizing both branches.")


def create_split(manifest_path: Path, output_path: Path, seed: int = 42) -> None:
    rows = [r for r in read_csv(manifest_path) if r["is_valid"] == "True" and r["is_in_clean_dataset"] == "True"]
    groups = defaultdict(list)
    for r in rows:
        near = r.get("near_duplicate_group_id", "")
        exact = r.get("duplicate_group_id", "")
        r["split_group_id"], r["split_group_source"] = (near, "near_duplicate") if near else ((exact, "exact_duplicate") if exact else (r["image_id"], "image_id"))
        groups[r["split_group_id"]].append(r)

    ratios = {"train": 4 / 6, "val": 1 / 6, "test": 1 / 6}
    output_rows = []
    by_label = defaultdict(list)
    for members in groups.values():
        by_label[members[0]["class_name"]].append(members)

    for label, label_groups in sorted(by_label.items()):
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(label_groups)
        label_groups.sort(key=len, reverse=True)
        targets = {name: len([r for group in label_groups for r in group]) * ratio for name, ratio in ratios.items()}
        counts = {name: 0 for name in ratios}
        priority = {"train": 2, "val": 1, "test": 0}
        for members in label_groups:
            split = max(ratios, key=lambda name: (targets[name] - counts[name], priority[name]))
            counts[split] += len(members)
            for r in members:
                output_rows.append({
                    "image_id": r["image_id"], "relative_path": r["relative_path"],
                    "class_name": label, "split": split,
                    "split_group_id": r["split_group_id"], "split_group_source": r["split_group_source"],
                    "sha256": r["sha256"],
                })

    output_rows.sort(key=lambda r: r["image_id"])
    write_csv(output_path, ["image_id", "relative_path", "class_name", "split", "split_group_id", "split_group_source", "sha256"], output_rows)
    print(f"Created split: {len(output_rows)} samples mapped to train/val/test.")


def extract_hog_branch(branch: str, dataset_root: Path, split_path: Path) -> Path:
    resized_root = dataset_root / "resized" / branch
    resized_manifest = resized_root / "resized_manifest.csv"
    output_npz = dataset_root / "features" / f"hog_{branch}.npz"
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    splits = {r["image_id"]: r for r in read_csv(split_path)}
    rows = [r for r in read_csv(resized_manifest) if r["is_resize_success"] == "True" and r["image_id"] in splits]
    classes = sorted({r["class_name"] for r in rows})
    labels = {c: i for i, c in enumerate(classes)}

    probe = hog(np.zeros((224, 224), dtype=np.uint8), **HOG_CONFIG)
    temp = Path(tempfile.mkdtemp(prefix=f"hog_{branch}_", dir=output_npz.parent))
    memmap_arr = np.memmap(temp / "X.dat", mode="w+", dtype=np.float32, shape=(len(rows), len(probe)))
    metadata = {k: [] for k in ("y", "split", "image_id", "relative_path", "sha256", "duplicate_group_id", "duplicate_representative_image_id")}

    print(f"Extracting HOG ({len(probe)} dims) for branch: {branch}...")
    try:
        for i, r in enumerate(rows):
            img = cv2.imread(str(resized_root / r["resized_relative_path"]), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            memmap_arr[i] = hog(gray, **HOG_CONFIG).astype(np.float32)
            s = splits[r["image_id"]]
            metadata["y"].append(labels[r["class_name"]])
            metadata["split"].append(s["split"])
            for k in ("image_id", "relative_path", "sha256", "duplicate_group_id", "duplicate_representative_image_id"):
                metadata[k].append(r.get(k, ""))
            if (i + 1) % 2000 == 0 or (i + 1) == len(rows):
                print(f"HOG [{branch}] {i + 1}/{len(rows)}", flush=True)

        memmap_arr.flush()
        np.savez(
            output_npz,
            X=memmap_arr,
            y=np.asarray(metadata["y"], dtype=np.int64),
            split=np.asarray(metadata["split"]),
            class_names=np.asarray(classes),
            image_ids=np.asarray(metadata["image_id"]),
            relative_paths=np.asarray(metadata["relative_path"]),
            sha256=np.asarray(metadata["sha256"]),
            duplicate_group_id=np.asarray(metadata["duplicate_group_id"]),
            duplicate_representative_image_id=np.asarray(metadata["duplicate_representative_image_id"]),
            hog_config_json=np.asarray(json.dumps(HOG_CONFIG, sort_keys=True)),
        )
    finally:
        del memmap_arr
        shutil.rmtree(temp, ignore_errors=True)

    print(f"Saved {output_npz} ({output_npz.stat().st_size / 1e6:.1f} MB)")
    return output_npz


def plot_cm(cm: np.ndarray, classes: list[str], title: str, output_path: Path) -> None:
    cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    cax = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm_display[i, j]:.1%}",
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def train_and_eval_svm(dataset_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_per_class = []

    for branch in ["direct", "padding"]:
        npz_file = dataset_root / "features" / f"hog_{branch}.npz"
        print("=" * 45)
        print(f" TRAINING SVM ON BRANCH: {branch.upper()}")
        print("=" * 45)

        with np.load(npz_file, allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]
            split = data["split"].astype(str)
            class_names = [str(c) for c in data["class_names"]]
            image_ids = [str(id_) for id_ in data["image_ids"]]
            relative_paths = [str(p) for p in data["relative_paths"]]

        tr_m, va_m, te_m = (split == "train"), (split == "val"), (split == "test")
        X_train, y_train = X[tr_m], y[tr_m]
        X_val, y_val = X[va_m], y[va_m]
        X_test, y_test = X[te_m], y[te_m]
        test_ids = [image_ids[i] for i, m in enumerate(te_m) if m]
        test_paths = [relative_paths[i] for i, m in enumerate(te_m) if m]

        print(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)} | Dim={X.shape[1]}")

        scaler = StandardScaler(with_mean=False)
        X_tr_sc = scaler.fit_transform(X_train)
        X_va_sc = scaler.transform(X_val)
        X_te_sc = scaler.transform(X_test)

        # 1. LinearSVC Tuning
        c_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        tuning_records = []
        best_c = 1.0
        best_val_f1 = -1.0

        print("Tuning LinearSVC parameter C on Validation set...")
        for c_val in c_candidates:
            t0 = time.time()
            clf = LinearSVC(C=c_val, class_weight="balanced", random_state=42, max_iter=5000, dual="auto")
            clf.fit(X_tr_sc, y_train)
            dur = time.time() - t0
            val_pred = clf.predict(X_va_sc)
            v_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            v_acc = accuracy_score(y_val, val_pred)
            tuning_records.append({"C": c_val, "val_macro_f1": round(v_f1, 4), "val_accuracy": round(v_acc, 4), "train_time": round(dur, 2)})
            print(f"  C={c_val:<7} -> Val Macro-F1: {v_f1:.4f} | Val Acc: {v_acc:.4f} ({dur:.1f}s)")
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_c = c_val

        pd.DataFrame(tuning_records).to_csv(output_dir / f"tuning_linearsvc_{branch}.csv", index=False)

        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
        ax.plot([r["C"] for r in tuning_records], [r["val_macro_f1"] for r in tuning_records], marker="o", lw=2, label="Validation Macro-F1")
        ax.plot([r["C"] for r in tuning_records], [r["val_accuracy"] for r in tuning_records], marker="s", lw=2, linestyle="--", label="Validation Accuracy")
        ax.set_xscale("log")
        ax.set_xlabel("Regularization C (log scale)")
        ax.set_ylabel("Score")
        ax.set_title(f"LinearSVC Tuning Curve ({branch.title()})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"val_curve_linearsvc_{branch}.png")
        plt.close(fig)

        print(f"Fitting optimal LinearSVC (C={best_c}) on Train...")
        t0 = time.time()
        final_lin = LinearSVC(C=best_c, class_weight="balanced", random_state=42, max_iter=5000, dual="auto")
        final_lin.fit(X_tr_sc, y_train)
        lin_time = time.time() - t0

        lin_pipe = Pipeline([("scaler", scaler), ("clf", final_lin)])
        joblib.dump(lin_pipe, output_dir / f"linearsvc_{branch}_model.joblib")

        te_pred = final_lin.predict(X_te_sc)
        all_summaries.append({
            "model": "LinearSVC", "branch": branch, "split": "test",
            "best_hyperparams": f"C={best_c}",
            "accuracy": round(accuracy_score(y_test, te_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, te_pred), 4),
            "macro_precision": round(precision_score(y_test, te_pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y_test, te_pred, average="macro", zero_division=0), 4),
            "macro_f1": round(f1_score(y_test, te_pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(y_test, te_pred, average="weighted", zero_division=0), 4),
            "train_time_sec": round(lin_time, 2),
        })

        rep = classification_report(y_test, te_pred, target_names=class_names, output_dict=True, zero_division=0)
        for cname in class_names:
            all_per_class.append({
                "model": "LinearSVC", "branch": branch, "class_name": cname,
                "precision": round(rep[cname]["precision"], 4),
                "recall": round(rep[cname]["recall"], 4),
                "f1_score": round(rep[cname]["f1-score"], 4),
                "support": rep[cname]["support"],
            })

        cm = confusion_matrix(y_test, te_pred)
        plot_cm(cm, class_names, f"LinearSVC Confusion Matrix ({branch.title()})", output_dir / f"confusion_matrix_linear_{branch}.png")

        misc = []
        for tid, rpath, y_t, y_p in zip(test_ids, test_paths, y_test, te_pred):
            if y_t != y_p:
                misc.append({"image_id": tid, "relative_path": rpath, "true_class": class_names[y_t], "pred_class": class_names[y_p]})
        pd.DataFrame(misc).to_csv(output_dir / f"misclassifications_linear_{branch}.csv", index=False)

        # 2. RBF Kernel with PCA
        print("Applying PCA (256 components) for RBF Kernel...")
        pca = PCA(n_components=256, random_state=42, svd_solver="randomized")
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_va_pca = pca.transform(X_va_sc)
        X_te_pca = pca.transform(X_te_sc)
        var_ratio = float(np.sum(pca.explained_variance_ratio_))
        print(f"Explained variance ratio by 256 PCs: {var_ratio:.2%}")

        rbf_c_candidates = [1.0, 10.0, 50.0]
        rbf_g_candidates = ["scale", 0.01, 0.001]
        best_rbf_p = (10.0, "scale")
        best_rbf_f1 = -1.0
        rbf_records = []

        print("Tuning RBF SVC on Validation set...")
        for c_rbf in rbf_c_candidates:
            for g_rbf in rbf_g_candidates:
                t0 = time.time()
                r_clf = SVC(C=c_rbf, gamma=g_rbf, kernel="rbf", class_weight="balanced", random_state=42)
                r_clf.fit(X_tr_pca, y_train)
                dur = time.time() - t0
                r_v_pred = r_clf.predict(X_va_pca)
                v_f1 = f1_score(y_val, r_v_pred, average="macro", zero_division=0)
                v_acc = accuracy_score(y_val, r_v_pred)
                rbf_records.append({"C": c_rbf, "gamma": str(g_rbf), "val_macro_f1": round(v_f1, 4), "val_accuracy": round(v_acc, 4), "train_time": round(dur, 2)})
                print(f"  RBF C={c_rbf}, gamma={g_rbf} -> Val Macro-F1: {v_f1:.4f} | Val Acc: {v_acc:.4f} ({dur:.1f}s)")
                if v_f1 > best_rbf_f1:
                    best_rbf_f1 = v_f1
                    best_rbf_p = (c_rbf, g_rbf)

        pd.DataFrame(rbf_records).to_csv(output_dir / f"tuning_rbf_{branch}.csv", index=False)

        b_c, b_g = best_rbf_p
        print(f"Fitting optimal RBF SVC (C={b_c}, gamma={b_g}) on Train...")
        t0 = time.time()
        final_rbf = SVC(C=b_c, gamma=b_g, kernel="rbf", class_weight="balanced", random_state=42)
        final_rbf.fit(X_tr_pca, y_train)
        rbf_time = time.time() - t0

        rbf_pipe = Pipeline([("scaler", scaler), ("pca", pca), ("clf", final_rbf)])
        joblib.dump(rbf_pipe, output_dir / f"rbf_svc_{branch}_model.joblib")

        rbf_te_pred = final_rbf.predict(X_te_pca)
        all_summaries.append({
            "model": "SVC(RBF+PCA)", "branch": branch, "split": "test",
            "best_hyperparams": f"C={b_c},gamma={b_g}",
            "accuracy": round(accuracy_score(y_test, rbf_te_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, rbf_te_pred), 4),
            "macro_precision": round(precision_score(y_test, rbf_te_pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y_test, rbf_te_pred, average="macro", zero_division=0), 4),
            "macro_f1": round(f1_score(y_test, rbf_te_pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(y_test, rbf_te_pred, average="weighted", zero_division=0), 4),
            "train_time_sec": round(rbf_time, 2),
        })

        rep_rbf = classification_report(y_test, rbf_te_pred, target_names=class_names, output_dict=True, zero_division=0)
        for cname in class_names:
            all_per_class.append({
                "model": "SVC(RBF+PCA)", "branch": branch, "class_name": cname,
                "precision": round(rep_rbf[cname]["precision"], 4),
                "recall": round(rep_rbf[cname]["recall"], 4),
                "f1_score": round(rep_rbf[cname]["f1-score"], 4),
                "support": rep_rbf[cname]["support"],
            })

        cm_rbf = confusion_matrix(y_test, rbf_te_pred)
        plot_cm(cm_rbf, class_names, f"RBF SVC Confusion Matrix ({branch.title()})", output_dir / f"confusion_matrix_rbf_{branch}.png")

    sum_df = pd.DataFrame(all_summaries)
    sum_df.to_csv(output_dir / "svm_metrics_summary.csv", index=False)
    pc_df = pd.DataFrame(all_per_class)
    pc_df.to_csv(output_dir / "svm_per_class_report.csv", index=False)

    print("================ FINAL TEST BENCHMARK ================")
    print(sum_df.to_string(index=False))


def main() -> int:
    t_start = time.time()
    work_dir = Path("/kaggle/working")
    raw_root = find_raw_dataset_root()

    manifest_file = unpack_clean_manifest(work_dir / "clean_manifest.csv")
    dataset_dir = work_dir / "dataset"
    output_dir = work_dir / "outputs" / "svm"

    print("Step 1/4: Resizing images (direct & padding)...")
    build_resized_branches(raw_root, manifest_file, dataset_dir)

    print("Step 2/4: Creating stratified 4/1/1 split...")
    split_file = dataset_dir / "split.csv"
    create_split(manifest_file, split_file, seed=42)

    print("Step 3/4: Extracting HOG features...")
    extract_hog_branch("direct", dataset_dir, split_file)
    extract_hog_branch("padding", dataset_dir, split_file)

    print("Step 4/4: Benchmarking SVM models...")
    train_and_eval_svm(dataset_dir, output_dir)

    print("Cleaning up large intermediate files...")
    shutil.rmtree(dataset_dir, ignore_errors=True)
    if manifest_file.exists():
        try:
            manifest_file.unlink()
        except Exception:
            pass

    total_time = time.time() - t_start
    print(f"Pipeline finished successfully in {total_time / 60:.2f} minutes!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')

Path("kaggle_kernel/main.py").write_text("".join(parts), encoding="utf-8")
print("Successfully generated kaggle_kernel/main.py")
