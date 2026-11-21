"""Streamlit application for the image classification demonstration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
HOG_CONFIG: dict[str, Any] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True,
}
MODEL_PATHS = {
    "SVM RBF": ROOT / "outputs" / "svm" / "rbf_svc_direct_model.joblib",
    "SVM tuyến tính": ROOT / "outputs" / "svm" / "linearsvc_direct_model.joblib",
    "Hồi quy logistic": ROOT / "outputs" / "logistic_regression" / "lr_padding_model.joblib",
}
# Resize branch each model was trained on (see scripts/build_resized_branches.py).
MODEL_BRANCHES = {
    "SVM RBF": "direct",
    "SVM tuyến tính": "direct",
    "Hồi quy logistic": "padding",
}
# Record field that stores each model's prediction for the session evaluation.
PREDICTION_FIELDS = {
    "SVM RBF": "rbf_index",
    "SVM tuyến tính": "linear_index",
    "Hồi quy logistic": "lr_index",
}
PADDING_BGR = (128, 128, 128)
CLASS_NAMES = ("billete", "knife", "monedero", "pistol", "smartphone", "tarjeta")
CLASS_NAME_BY_INDEX = dict(enumerate(CLASS_NAMES))
SESSION_RECORDS = "evaluation_records"


if SESSION_RECORDS not in st.session_state:
    st.session_state[SESSION_RECORDS] = []


@st.cache_resource
def load_pipeline(model_path: str) -> Any:
    """Load a persisted scikit-learn pipeline once per process."""
    return joblib.load(model_path)


def decode_image(uploaded_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV BGR image."""
    encoded = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể giải mã ảnh. Vui lòng chọn tệp ảnh hợp lệ.")
    return image


def resize_padding(img_bgr: np.ndarray, interpolation: int) -> np.ndarray:
    """Keep the aspect ratio, then pad with neutral gray to 224 x 224."""
    target = IMAGE_SIZE[0]
    height, width = img_bgr.shape[:2]
    scale = min(target / width, target / height)
    new_w = max(1, min(target, round(width * scale)))
    new_h = max(1, min(target, round(height * scale)))
    if (new_w, new_h) != (width, height):
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)
    left, top = (target - new_w) // 2, (target - new_h) // 2
    return cv2.copyMakeBorder(
        img_bgr, top, target - new_h - top, left, target - new_w - left,
        cv2.BORDER_CONSTANT, value=PADDING_BGR,
    )


def extract_features(img_bgr: np.ndarray, branch: str = "direct") -> np.ndarray:
    """Apply the given resize branch and return one HOG feature vector."""
    height, width = img_bgr.shape[:2]
    # Same rule as scripts/build_resized_branches.py.
    interpolation = cv2.INTER_CUBIC if width < 224 and height < 224 else cv2.INTER_AREA
    if branch == "padding":
        resized = resize_padding(img_bgr, interpolation)
    else:
        resized = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=interpolation)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, **HOG_CONFIG).astype(np.float32).reshape(1, -1)
    if features.shape[1] != 26244:
        raise RuntimeError(
            f"Cấu hình HOG tạo {features.shape[1]} đặc trưng, cần đúng 26.244."
        )
    return features


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert relative decision scores to normalized display scores."""
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def predict(
    img_bgr: np.ndarray, pipeline: Any, branch: str = "direct"
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Predict a class and return its label, display scores, and class label.

    Logistic regression returns its own predict_proba; SVM decision scores are
    turned into relative scores with softmax.
    """
    features = extract_features(img_bgr, branch)
    prediction = pipeline.predict(features)[0]
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(features)[0], dtype=np.float64)
    else:
        raw_scores = np.asarray(pipeline.decision_function(features))
        scores = raw_scores[0] if raw_scores.ndim > 1 else raw_scores
        probabilities = softmax(np.asarray(scores, dtype=np.float64))
    classes = np.asarray(getattr(pipeline, "classes_", np.arange(len(probabilities))))
    class_index = int(np.flatnonzero(classes == prediction)[0])
    return prediction, probabilities, classes[class_index]


def class_names(pipeline: Any, count: int) -> list[str]:
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        classes = list(range(count))
    return [CLASS_NAME_BY_INDEX.get(int(value), str(value)) for value in classes]


def render_prediction_tab() -> None:
    st.subheader("Thử nghiệm dự đoán")
    model_name = st.selectbox("Mô hình", list(MODEL_PATHS))
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        st.warning(
            f"Chưa tìm thấy `{model_path.name}`. Hãy đặt pipeline đã huấn luyện vào thư mục ứng dụng."
        )
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        return
    st.image(uploaded_file, caption="Ảnh đầu vào", use_container_width=True)
    if not model_path.exists():
        return
    try:
        image = decode_image(uploaded_file.getvalue())
        pipeline = load_pipeline(str(model_path))
        prediction, probabilities, _ = predict(image, pipeline, MODEL_BRANCHES[model_name])
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        st.error(f"Không thể dự đoán: {error}")
        return

    labels = class_names(pipeline, len(probabilities))
    predicted_label = str(prediction)
    try:
        predicted_name = CLASS_NAME_BY_INDEX[int(prediction)]
    except (KeyError, ValueError, TypeError):
        predicted_name = predicted_label

    linear_pipeline = load_pipeline(str(MODEL_PATHS["SVM tuyến tính"]))
    linear_features = extract_features(image)
    linear_prediction = linear_pipeline.predict(linear_features)[0]
    linear_name = CLASS_NAME_BY_INDEX.get(int(linear_prediction), str(linear_prediction))
    rbf_prediction = prediction if model_name == "SVM RBF" else None
    if rbf_prediction is None:
        rbf_pipeline = load_pipeline(str(MODEL_PATHS["SVM RBF"]))
        rbf_prediction = rbf_pipeline.predict(linear_features)[0]
    rbf_name = CLASS_NAME_BY_INDEX.get(int(rbf_prediction), str(rbf_prediction))
    lr_prediction = prediction if model_name == "Hồi quy logistic" else None
    if lr_prediction is None:
        lr_pipeline = load_pipeline(str(MODEL_PATHS["Hồi quy logistic"]))
        lr_prediction = lr_pipeline.predict(extract_features(image, "padding"))[0]
    lr_name = CLASS_NAME_BY_INDEX.get(int(lr_prediction), str(lr_prediction))

    with st.container(border=True):
        st.markdown("### Nhãn đúng của ảnh là")
        st.caption("Chọn nhãn chính xác để tự động cập nhật báo cáo đánh giá và phân tích lỗi.")
        true_name = st.selectbox(
            "Chọn nhãn đúng",
            CLASS_NAMES,
            index=int(prediction) if int(prediction) in range(len(CLASS_NAMES)) else 0,
        )
    record_key = (
        str(uploaded_file.name),
        hash(uploaded_file.getvalue()),
        model_name,
    )
    record = {
        "key": record_key,
        "name": uploaded_file.name,
        "image": uploaded_file.getvalue(),
        "features": linear_features,
        "true_name": true_name,
        "true_index": CLASS_NAMES.index(true_name),
        "predicted_name": predicted_name,
        "predicted_index": int(prediction),
        "linear_name": linear_name,
        "linear_index": int(linear_prediction),
        "rbf_name": rbf_name,
        "rbf_index": int(rbf_prediction),
        "lr_name": lr_name,
        "lr_index": int(lr_prediction),
    }
    matching_index = next(
        (
            index
            for index, existing in enumerate(st.session_state[SESSION_RECORDS])
            if existing.get("key") == record_key
        ),
        None,
    )
    if matching_index is None:
        st.session_state[SESSION_RECORDS].append(record)
    else:
        st.session_state[SESSION_RECORDS][matching_index] = record

    st.divider()
    is_weapon = any(
        name in predicted_name.lower()
        for name in ("knife", "pistol")
    )
    color = "#d62728" if is_weapon else "#16803c"
    st.markdown(
        f'<h3 style="color:{color}">Nhãn dự đoán: {predicted_name}</h3>',
        unsafe_allow_html=True,
    )
    if model_name == "Hồi quy logistic":
        st.caption("Xác suất do hồi quy logistic ước lượng (predict_proba).")
    else:
        st.caption("Điểm số đã chuẩn hóa bằng softmax từ decision_function; không phải xác suất hiệu chuẩn.")
    st.bar_chart(dict(zip(labels, probabilities)))
    st.dataframe(
        {"Lớp": labels, "Điểm số": probabilities},
        hide_index=True,
        use_container_width=True,
    )


def render_evaluation_tab() -> None:
    st.subheader("Báo cáo đánh giá")
    model_name = st.selectbox(
        "Mô hình đánh giá",
        list(MODEL_PATHS),
        key="evaluation_model",
    )
    records = st.session_state[SESSION_RECORDS]
    if not records:
        st.info("Tải ảnh và chọn nhãn đúng trong tab dự đoán để bắt đầu tạo báo cáo.")
        return

    prediction_field = PREDICTION_FIELDS[model_name]
    true_indices = np.asarray([record["true_index"] for record in records])
    predicted_indices = np.asarray([record[prediction_field] for record in records])
    all_indices = list(range(len(CLASS_NAMES)))

    col1, col2, col3 = st.columns(3)
    col1.metric("Số mẫu", len(records))
    col2.metric("Accuracy", f"{accuracy_score(true_indices, predicted_indices):.3f}")
    col3.metric(
        "Macro F1",
        f"{f1_score(true_indices, predicted_indices, average='macro', labels=all_indices, zero_division=0):.3f}",
    )

    report = classification_report(
        true_indices,
        predicted_indices,
        labels=all_indices,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_rows = [
        {
            "Lớp": name,
            "Precision": report[name]["precision"],
            "Recall": report[name]["recall"],
            "F1-score": report[name]["f1-score"],
            "Số mẫu": report[name]["support"],
        }
        for name in CLASS_NAMES
    ]
    st.subheader("Chỉ số theo từng lớp")
    st.dataframe(report_rows, hide_index=True, use_container_width=True)

    matrix = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=all_indices,
    )
    st.subheader("Ma trận nhầm lẫn")
    st.dataframe(
        {
            "Nhãn đúng \\ Dự đoán": CLASS_NAMES,
            **{name: matrix[:, index].tolist() for index, name in enumerate(CLASS_NAMES)},
        },
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Đường cong tinh chỉnh C")
    if len(records) >= 4 and len(np.unique(true_indices)) >= 2:
        features = np.vstack([record["features"] for record in records])
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        curve_points: dict[str, float] = {}
        cv = LeaveOneOut()
        for c_value in c_values:
            tuned_model = make_pipeline(
                StandardScaler(),
                LinearSVC(C=c_value, dual="auto", random_state=42),
            )
            try:
                scores = cross_val_score(
                    tuned_model,
                    features,
                    true_indices,
                    cv=cv,
                    scoring="accuracy",
                )
                curve_points[str(c_value)] = float(np.mean(scores))
            except ValueError:
                curve_points[str(c_value)] = 0.0
        st.line_chart(curve_points)
    else:
        st.caption(
            "Cần ít nhất 4 mẫu và 2 lớp để vẽ đường cong tinh chỉnh C."
        )


def render_error_tab() -> None:
    st.subheader("Phân tích lỗi")
    records = st.session_state[SESSION_RECORDS]
    if not records:
        st.info("Tải ảnh và chọn nhãn đúng trong tab dự đoán để bắt đầu phân tích lỗi.")
        return

    misclassified = [
        record
        for record in records
        if record["linear_index"] != record["true_index"]
    ]
    st.metric("Số ảnh SVM tuyến tính phân loại sai", len(misclassified))

    if misclassified:
        for record in misclassified:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(record["image"], caption=record["name"], use_container_width=True)
            with col2:
                st.write(f"**Nhãn đúng:** {record['true_name']}")
                st.write(f"**SVM tuyến tính dự đoán:** {record['linear_name']}")
        confusion_pairs: dict[tuple[str, str], int] = {}
        for record in misclassified:
            pair = (record["true_name"], record["linear_name"])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        st.subheader("Các cặp lớp thường bị nhầm lẫn")
        st.dataframe(
            [
                {
                    "Nhãn đúng": true_name,
                    "Dự đoán sai": predicted_name,
                    "Số lần": count,
                }
                for (true_name, predicted_name), count in sorted(
                    confusion_pairs.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("SVM tuyến tính chưa phân loại sai mẫu nào trong phiên hiện tại.")


def main() -> None:
    st.set_page_config(page_title="Phân loại ảnh", page_icon="🔎", layout="wide")
    st.title("Phân loại ảnh bằng HOG và SVM")
    st.caption(
        "Hai mô hình SVM co giãn ảnh trực tiếp về 224 × 224; hồi quy logistic giữ tỷ lệ "
        "và đệm viền xám. Sau đó ứng dụng trích xuất HOG."
    )
    prediction_tab, evaluation_tab, error_tab = st.tabs(
        ["Thử nghiệm dự đoán", "Báo cáo đánh giá", "Phân tích lỗi"]
    )
    with prediction_tab:
        render_prediction_tab()
    with evaluation_tab:
        render_evaluation_tab()
    with error_tab:
        render_error_tab()


if __name__ == "__main__":
    main()