#!/usr/bin/env python3
"""Streamlit web dashboard for SOHAS Weapon Classification SVM model."""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from skimage.feature import hog
import streamlit as st

st.set_page_config(
    page_title="SOHAS Weapon Classification - SVM Dashboard",
    page_icon="🛡️",
    layout="wide",
)

CLASSES = ["billete", "knife", "monedero", "pistol", "smartphone", "tarjeta"]
CLASS_INFO = {
    "knife": {"vi": "Dao / Vũ khí lạnh", "type": "VŨ KHÍ NGUY HIỂM", "is_weapon": True},
    "pistol": {"vi": "Súng ngắn / Vũ khí nóng", "type": "VŨ KHÍ NGUY HIỂM", "is_weapon": True},
    "smartphone": {"vi": "Điện thoại", "type": "Đồ vật thông thường", "is_weapon": False},
    "tarjeta": {"vi": "Thẻ ATM / Thẻ từ", "type": "Đồ vật thông thường", "is_weapon": False},
    "billete": {"vi": "Tiền giấy", "type": "Đồ vật thông thường", "is_weapon": False},
    "monedero": {"vi": "Ví tiền / Bóp", "type": "Đồ vật thông thường", "is_weapon": False},
}

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


@st.cache_resource
def load_pipeline(model_path: str):
    return joblib.load(model_path)


def predict(img_bgr: np.ndarray, pipeline) -> dict:
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_AREA if (w > 224 or h > 224) else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (224, 224), interpolation=interp)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    feat = hog(gray, **HOG_CONFIG).astype(np.float32).reshape(1, -1)

    pred_idx = int(pipeline.predict(feat)[0])
    scores = pipeline.decision_function(feat)[0]
    probs = softmax(scores)

    results = []
    for i, cname in enumerate(CLASSES):
        results.append({
            "class": cname,
            "name_vi": CLASS_INFO[cname]["vi"],
            "type": CLASS_INFO[cname]["type"],
            "is_weapon": CLASS_INFO[cname]["is_weapon"],
            "score": float(scores[i]),
            "prob": float(probs[i]),
        })
    results.sort(key=lambda x: x["prob"], reverse=True)
    return {
        "pred_class": CLASSES[pred_idx],
        "top": results[0],
        "all": results,
    }


def main():
    st.title("🛡️ SOHAS Weapon Classification - SVM Dashboard")
    st.markdown("**BTL Machine Learning (CO3117)** - Phân loại vũ khí và đồ vật bằng **HOG + PCA + SVM**")

    st.sidebar.header("Cấu hình")
    model_choice = st.sidebar.selectbox(
        "Chọn mô hình SVM:",
        [
            ("RBF SVC + PCA (Khuyên dùng)", "outputs/svm/rbf_svc_direct_model.joblib"),
            ("LinearSVC (Baseline)", "outputs/svm/linearsvc_direct_model.joblib"),
        ],
        format_func=lambda x: x[0],
    )
    model_path = model_choice[1]

    if not Path(model_path).exists():
        st.error(f"Không tìm thấy file mô hình tại `{model_path}`.")
        return

    pipeline = load_pipeline(model_path)

    tab1, tab2, tab3 = st.tabs(["🔍 Thử nghiệm suy luận", "📊 Báo cáo đánh giá (Test Set)", "⚠️ Phân tích lỗi (Error Analysis)"])

    # TAB 1: Inference
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Ảnh đầu vào")
            uploaded = st.file_uploader("Upload file ảnh (JPG, PNG):", type=["jpg", "jpeg", "png"])
            
            img_bgr = None
            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=uploaded.name, use_container_width=True)
            else:
                st.info("Hãy upload ảnh để bắt đầu phân loại.")

        with col2:
            st.subheader("2. Kết quả nhận diện")
            if img_bgr is not None:
                with st.spinner("Đang trích xuất HOG & chạy SVM..."):
                    res = predict(img_bgr, pipeline)
                    top = res["top"]

                if top["is_weapon"]:
                    st.error(f"🚨 **CẢNH BÁO: PHÁT HIỆN {top['name_vi'].upper()} ({top['class'].upper()})**")
                else:
                    st.success(f"✅ **AN TOÀN: {top['name_vi'].upper()} ({top['class'].upper()})**")

                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Vật thể dự đoán", f"{top['class']} ({top['name_vi']})")
                col_m2.metric("Độ tin cậy", f"{top['prob']*100:.2f}%")

                st.write("**Xác suất các lớp (Softmax trên Decision Margin):**")
                chart_df = pd.DataFrame([
                    {"Lớp": f"{r['class']} ({r['name_vi']})", "Xác suất (%)": round(r['prob'] * 100, 2)}
                    for r in res["all"]
                ]).set_index("Lớp")
                st.bar_chart(chart_df)
            else:
                st.info("Vui lòng chọn hoặc tải lên một ảnh để phân loại.")

    # TAB 2: Metrics Benchmark
    with tab2:
        st.subheader("Bảng so sánh hiệu năng các mô hình (Tập Test)")
        metrics_file = Path("outputs/svm/svm_metrics_summary.csv")
        if metrics_file.exists():
            df_metrics = pd.read_csv(metrics_file)
            st.dataframe(df_metrics, use_container_width=True)
        else:
            st.warning("Chưa có file svm_metrics_summary.csv")

        st.divider()
        st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Mô hình RBF SVC + PCA (Direct)**")
            p1 = Path("outputs/svm/confusion_matrix_rbf_direct.png")
            if p1.exists():
                st.image(str(p1), use_container_width=True)
        with c2:
            st.write("**Mô hình LinearSVC (Direct)**")
            p2 = Path("outputs/svm/confusion_matrix_linear_direct.png")
            if p2.exists():
                st.image(str(p2), use_container_width=True)

        st.divider()
        st.subheader("Đường cong tinh chỉnh siêu tham số (Hyperparameter Tuning)")
        p3 = Path("outputs/svm/val_curve_linearsvc_direct.png")
        if p3.exists():
            st.image(str(p3), caption="Quét siêu tham số C cho LinearSVC", use_container_width=True)

    # TAB 3: Misclassifications
    with tab3:
        st.subheader("Phân tích lỗi nhầm lẫn (Misclassifications)")
        misc_file = Path("outputs/svm/misclassifications_linear_direct.csv")
        if misc_file.exists():
            df_misc = pd.read_csv(misc_file)
            st.write(f"Tổng số mẫu đoán sai trên Test set (LinearSVC): **{len(df_misc)} / 1588 mẫu**")

            st.write("**Top các cặp lớp bị nhầm lẫn nhiều nhất:**")
            counts = df_misc.groupby(["true_class", "pred_class"]).size().reset_index(name="count")
            counts = counts.sort_values(by="count", ascending=False)
            st.dataframe(counts, use_container_width=True)

            st.write("**Danh sách chi tiết các mẫu bị dự đoán sai:**")
            st.dataframe(df_misc, use_container_width=True)
        else:
            st.warning("Chưa có file misclassifications.")


if __name__ == "__main__":
    main()
