#!/usr/bin/env python3
"""Train, tune, evaluate, and analyze SVM models on extracted SOHAS HOG features."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("dataset/features"),
        help="Directory containing hog_direct.npz and hog_padding.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/svm"),
        help="Directory where evaluation metrics, tables, and figures will be saved.",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["direct", "padding"],
        choices=["direct", "padding"],
        help="Which feature branches to evaluate.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=256,
        help="Number of PCA components for RBF Kernel SVM.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_branch_data(npz_path: Path) -> dict[str, Any]:
    print(f"Loading features from {npz_path}...")
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "X": data["X"],
            "y": data["y"],
            "split": data["split"].astype(str),
            "class_names": [str(c) for c in data["class_names"]],
            "image_ids": [str(id_) for id_ in data["image_ids"]],
            "relative_paths": [str(p) for p in data["relative_paths"]],
        }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str,
    output_path: Path,
    normalize: bool = True,
) -> None:
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
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
            val_str = f"{cm_display[i, j]:{fmt}}"
            ax.text(
                j,
                i,
                val_str,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved confusion matrix plot: {output_path}")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    branch_name: str,
    split_name: str,
    training_time_sec: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    summary = {
        "model": model_name,
        "branch": branch_name,
        "split": split_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "training_time_sec": round(training_time_sec, 2),
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cname in class_names:
        row = report_dict[cname]
        rows.append({
            "model": model_name,
            "branch": branch_name,
            "split": split_name,
            "class_name": cname,
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1_score": round(row["f1-score"], 4),
            "support": int(row["support"]),
        })
    per_class_df = pd.DataFrame(rows)
    return summary, per_class_df


def tune_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_candidates: list[float],
    class_weight: str = "balanced",
    random_state: int = 42,
) -> tuple[float, list[dict[str, Any]]]:
    tuning_history = []
    scaler = StandardScaler(with_mean=False)
    X_tr_scaled = scaler.fit_transform(X_train)
    X_va_scaled = scaler.transform(X_val)

    best_c = c_candidates[0]
    best_val_f1 = -1.0

    print("--- Tuning LinearSVC on Validation Set ---")
    for c_val in c_candidates:
        t0 = time.time()
        clf = LinearSVC(
            C=c_val,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=5000,
            dual="auto",
        )
        clf.fit(X_tr_scaled, y_train)
        train_time = time.time() - t0

        val_pred = clf.predict(X_va_scaled)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred)

        record = {
            "C": c_val,
            "val_macro_f1": round(val_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "train_time_sec": round(train_time, 2),
        }
        tuning_history.append(record)
        print(f"C={c_val:<7} -> Val Macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} ({train_time:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c_val

    print(f"Optimal C selected: {best_c} (Val Macro-F1: {best_val_f1:.4f})")
    return best_c, tuning_history


def tune_rbf_svm(
    X_train_pca: np.ndarray,
    y_train: np.ndarray,
    X_val_pca: np.ndarray,
    y_val: np.ndarray,
    c_candidates: list[float],
    gamma_candidates: list[str | float],
    class_weight: str = "balanced",
    random_state: int = 42,
) -> tuple[tuple[float, str | float], list[dict[str, Any]]]:
    tuning_history = []
    best_params = (c_candidates[0], gamma_candidates[0])
    best_val_f1 = -1.0

    print("--- Tuning RBF SVC on Validation Set ---")
    for c_val in c_candidates:
        for g_val in gamma_candidates:
            t0 = time.time()
            clf = SVC(
                C=c_val,
                gamma=g_val,
                kernel="rbf",
                class_weight=class_weight,
                random_state=random_state,
            )
            clf.fit(X_train_pca, y_train)
            train_time = time.time() - t0

            val_pred = clf.predict(X_val_pca)
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_pred)

            record = {
                "C": c_val,
                "gamma": str(g_val),
                "val_macro_f1": round(val_f1, 4),
                "val_accuracy": round(val_acc, 4),
                "train_time_sec": round(train_time, 2),
            }
            tuning_history.append(record)
            print(f"C={c_val}, gamma={g_val} -> Val Macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} ({train_time:.1f}s)")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_params = (c_val, g_val)

    print(f"Optimal RBF Params: C={best_params[0]}, gamma={best_params[1]} (Val Macro-F1: {best_val_f1:.4f})")
    return best_params, tuning_history


def plot_tuning_curve(
    tuning_history: list[dict[str, Any]],
    output_path: Path,
    title: str = "LinearSVC Hyperparameter Tuning (C vs Validation Metrics)",
) -> None:
    c_values = [h["C"] for h in tuning_history]
    f1_values = [h["val_macro_f1"] for h in tuning_history]
    acc_values = [h["val_accuracy"] for h in tuning_history]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(c_values, f1_values, marker="o", color="#1f77b4", label="Validation Macro-F1", lw=2)
    ax.plot(c_values, acc_values, marker="s", color="#ff7f0e", linestyle="--", label="Validation Accuracy", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization Parameter C (log scale)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved tuning curve: {output_path}")


def analyze_misclassifications(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    image_ids: list[str],
    relative_paths: list[str],
    class_names: list[str],
    output_path: Path,
    limit: int = 50,
) -> None:
    records = []
    for true_idx, pred_idx, img_id, rel_p in zip(y_test, y_pred, image_ids, relative_paths):
        if true_idx != pred_idx:
            records.append({
                "image_id": img_id,
                "relative_path": rel_p,
                "true_class": class_names[true_idx],
                "predicted_class": class_names[pred_idx],
            })
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Found {len(records)} test misclassifications. Saved report: {output_path}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_per_class = []

    for branch in args.branches:
        npz_file = args.features_dir / f"hog_{branch}.npz"
        if not npz_file.exists():
            print(f"Error: Feature file not found: {npz_file}")
            continue

        print(f"\n==========================================")
        print(f" PROCESSING BRANCH: {branch.upper()}")
        print(f"==========================================")

        data = load_branch_data(npz_file)
        X, y, split = data["X"], data["y"], data["split"]
        class_names = data["class_names"]

        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        test_ids = [data["image_ids"][i] for i, m in enumerate(test_mask) if m]
        test_paths = [data["relative_paths"][i] for i, m in enumerate(test_mask) if m]

        print(f"Dataset partition counts: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
        print(f"Feature dimension: {X.shape[1]}")

        # -------------------------------------------------------------
        # 1. LinearSVC: Hyperparameter tuning on C
        # -------------------------------------------------------------
        c_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_c, lin_tuning_hist = tune_linear_svm(
            X_train, y_train, X_val, y_val, c_list, class_weight="balanced", random_state=args.random_state
        )

        pd.DataFrame(lin_tuning_hist).to_csv(
            args.output_dir / f"tuning_linearsvc_{branch}.csv", index=False
        )
        plot_tuning_curve(
            lin_tuning_hist,
            args.output_dir / f"val_curve_linearsvc_{branch}.png",
            title=f"LinearSVC Tuning (Branch: {branch})",
        )

        # Train final LinearSVC with best C on train set
        print(f"\nTraining optimal LinearSVC (C={best_c}) on Train set...")
        t0 = time.time()
        linear_pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(C=best_c, class_weight="balanced", random_state=args.random_state, max_iter=5000, dual="auto")),
        ])
        linear_pipeline.fit(X_train, y_train)
        lin_train_time = time.time() - t0

        # Evaluate LinearSVC on Validation and Test
        val_pred = linear_pipeline.predict(X_val)
        val_sum, val_pc = evaluate_predictions(y_val, val_pred, class_names, "LinearSVC", branch, "val", lin_train_time)
        all_summaries.append(val_sum)
        all_per_class.append(val_pc)

        test_pred = linear_pipeline.predict(X_test)
        test_sum, test_pc = evaluate_predictions(y_test, test_pred, class_names, "LinearSVC", branch, "test", lin_train_time)
        all_summaries.append(test_sum)
        all_per_class.append(test_pc)

        # Save model
        joblib.dump(linear_pipeline, args.output_dir / f"linear_svm_{branch}_model.joblib")

        # Confusion Matrix for LinearSVC
        cm_linear = confusion_matrix(y_test, test_pred)
        plot_confusion_matrix(
            cm_linear,
            class_names,
            f"Normalized Confusion Matrix - LinearSVC ({branch.title()})",
            args.output_dir / f"confusion_matrix_linear_{branch}.png",
        )

        # Error analysis
        analyze_misclassifications(
            y_test,
            test_pred,
            test_ids,
            test_paths,
            class_names,
            args.output_dir / f"misclassifications_linear_{branch}.csv",
        )

        # -------------------------------------------------------------
        # 2. Kernel RBF SVC with PCA reduction
        # -------------------------------------------------------------
        print(f"\nApplying PCA reduction to {args.pca_components} components for RBF Kernel...")
        pca_scaler = StandardScaler(with_mean=False)
        X_tr_sc = pca_scaler.fit_transform(X_train)
        X_va_sc = pca_scaler.transform(X_val)
        X_te_sc = pca_scaler.transform(X_test)

        pca = PCA(n_components=args.pca_components, random_state=args.random_state, svd_solver="randomized")
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_va_pca = pca.transform(X_va_sc)
        X_te_pca = pca.transform(X_te_sc)
        explained_var = float(np.sum(pca.explained_variance_ratio_))
        print(f"Cumulative explained variance by {args.pca_components} components: {explained_var:.2%}")

        rbf_c_list = [1.0, 10.0, 50.0]
        rbf_gamma_list = ["scale", 0.01, 0.001]
        best_rbf_params, rbf_tuning_hist = tune_rbf_svm(
            X_tr_pca, y_train, X_va_pca, y_val, rbf_c_list, rbf_gamma_list, class_weight="balanced", random_state=args.random_state
        )

        pd.DataFrame(rbf_tuning_hist).to_csv(
            args.output_dir / f"tuning_rbf_svc_{branch}.csv", index=False
        )

        # Train final RBF model
        best_rbf_c, best_rbf_g = best_rbf_params
        print(f"\nTraining optimal RBF SVC (C={best_rbf_c}, gamma={best_rbf_g}) on Train set...")
        t0 = time.time()
        rbf_clf = SVC(
            C=best_rbf_c,
            gamma=best_rbf_g,
            kernel="rbf",
            class_weight="balanced",
            random_state=args.random_state,
        )
        rbf_clf.fit(X_tr_pca, y_train)
        rbf_train_time = time.time() - t0

        rbf_val_pred = rbf_clf.predict(X_va_pca)
        rbf_val_sum, rbf_val_pc = evaluate_predictions(y_val, rbf_val_pred, class_names, "SVC(RBF+PCA)", branch, "val", rbf_train_time)
        all_summaries.append(rbf_val_sum)
        all_per_class.append(rbf_val_pc)

        rbf_test_pred = rbf_clf.predict(X_te_pca)
        rbf_test_sum, rbf_test_pc = evaluate_predictions(y_test, rbf_test_pred, class_names, "SVC(RBF+PCA)", branch, "test", rbf_train_time)
        all_summaries.append(rbf_test_sum)
        all_per_class.append(rbf_test_pc)

        # Confusion Matrix for RBF SVC
        cm_rbf = confusion_matrix(y_test, rbf_test_pred)
        plot_confusion_matrix(
            cm_rbf,
            class_names,
            f"Normalized Confusion Matrix - RBF SVC ({branch.title()})",
            args.output_dir / f"confusion_matrix_rbf_{branch}.png",
        )

    # -----------------------------------------------------------------
    # Save Combined Master Summary Tables
    # -----------------------------------------------------------------
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = args.output_dir / "svm_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n==========================================")
        print(f" ALL SVM EXPERIMENTS COMPLETED!")
        print(f" Saved metrics summary: {summary_path}")
        print(summary_df.to_string(index=False))

    if all_per_class:
        combined_pc = pd.concat(all_per_class, ignore_index=True)
        pc_path = args.output_dir / "svm_per_class_report.csv"
        combined_pc.to_csv(pc_path, index=False)
        print(f" Saved per-class report: {pc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
