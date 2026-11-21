# SOHAS Weapon Classification

Project đã chuẩn bị dữ liệu SOHAS theo luồng: Dataset Collection + Exploration
-> kiểm tra/làm sạch ảnh -> tạo hai nhánh resize -> chia train/validation/test -> trích xuất HOG.
(2 nhánh resize - 1 nhánh resize ảnh direct, 1 nhánh resize ảnh có padding (tùy model))
## 1. Cấu trúc thư mục

```text
app.py                         # ứng dụng web Streamlit (SVM RBF, SVM tuyến tính, hồi quy logistic)
Sohas_weapon-Classification/   # ảnh raw theo 6 thư mục lớp, không chỉnh sửa
outputs/data_cleaning/         # metadata, thống kê và cleaning của B1
scripts/                       # mã nguồn chạy pipeline
notebooks/                     # notebook Logistic Regression
dataset/                       # ảnh resize, split và HOG sinh ra khi chạy B2/B3
outputs/svm/                   # kết quả và model SVM
outputs/logistic_regression/   # kết quả Logistic Regression
report/                        # mã nguồn LaTeX của báo cáo
```

### Chức năng từng thư mục

- `Sohas_weapon-Classification/`: dataset gốc gồm 6 lớp `billete`, `knife`, `monedero`, `pistol`, `smartphone`, `tarjeta`.
- `outputs/data_cleaning/clean_manifest.csv` là nguồn quyết định ảnh nào được phép dùng tiếp (`is_in_clean_dataset=True`).
- `scripts/`: các script CLI; không sửa raw data.
- `dataset/original/`: symlink tới dataset gốc, không phải bản copy.
- `dataset/resized/direct/`: ảnh `224×224` resize trực tiếp, có manifest và log lỗi riêng.
- `dataset/resized/padding/`: ảnh giữ aspect ratio, padding xám trung tính `BGR=(128,128,128)`, có manifest và log lỗi riêng.
- `dataset/split.csv`: mapping từng ảnh clean sang `train`, `val` hoặc `test`. Đã lọc các ảnh trùng SHA256.
- `dataset/features/`: HOG cho hai nhánh.

## 2. Chức năng từng script

| Script | Chức năng |
|---|---|
| `build_image_manifest.py` | Quét dataset, xác minh decode ảnh và tạo `image_manifest.csv`/`corrupted_images.csv`. |
| `summarize_dataset.py` | Thống kê số ảnh, kích thước, aspect ratio và format theo lớp. |
| `create_b1_figures.py` | Vẽ phân bố lớp, ảnh mẫu và phân bố độ phân giải. |
| `find_exact_duplicates.py` | Tính SHA-256 theo chunk, phát hiện nhóm ảnh trùng hoàn toàn. |
| `build_clean_manifest.py` | Tạo `clean_manifest.csv`; giữ một đại diện khi ảnh trùng cùng nhãn và gắn cờ xung đột nhãn. |
| `build_resized_branches.py` | Tạo hai nhánh ảnh `direct` và `padding` ở `224×224`. |
| `create_split.py` | Chia stratified theo tỷ lệ `4/1/1`, seed 42, giữ các duplicate group trong cùng split. |
| `extract_hog_by_split.py` | Đọc ảnh đã resize và `split.csv`, sau đó tạo HOG `.npz` cho một nhánh. |
| `train_svm.py` | Tinh chỉnh, huấn luyện và đánh giá SVM tuyến tính và SVM RBF trên hai nhánh; lưu model và kết quả vào `outputs/svm/`. |
| `predict.py` | Dự đoán lớp cho một ảnh từ dòng lệnh bằng model SVM đã lưu. |

## 3. Cài đặt môi trường và tải dữ liệu

Sử dụng Python 3.12 để chạy tiền xử lý và huấn luyện trên CPU. Chạy các lệnh từ thư mục chứa `README.md`. Nếu chỉ muốn thử mô hình đã huấn luyện, xem mục 4.

Trên Linux hoặc macOS, tạo và kích hoạt môi trường ảo.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Trên Windows, chạy trong Command Prompt.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Sau khi kích hoạt môi trường ảo, cài thư viện cho quy trình huấn luyện.

```bash
python -m pip install -r requirements.txt
```

Tải và giải nén [SOHAS Classification trên Kaggle](https://www.kaggle.com/datasets/natansantosmoura/sohas-classification?resource=download-directory). Đặt sáu thư mục lớp trực tiếp trong `Sohas_weapon-Classification/` cạnh `README.md`.

```text
Sohas_weapon-Classification/
├── billete/
├── knife/
├── monedero/
├── pistol/
├── smartphone/
└── tarjeta/
```

## 4. Chạy dashboard với mô hình đã huấn luyện

Dashboard dùng ảnh bạn tải lên và ba file mô hình có sẵn trong thư mục bài làm.

- `outputs/svm/rbf_svc_direct_model.joblib`
- `outputs/svm/linearsvc_direct_model.joblib`
- `outputs/logistic_regression/lr_padding_model.joblib`

### Chạy bằng Docker

```bash
docker build -t sohas-weapon-classification .
docker run --rm -p 8501:8501 sohas-weapon-classification
```

Sau khi container khởi động, mở `http://localhost:8501`.

### Chạy bằng Python

Tạo môi trường ảo Python 3.12 theo mục 3, dùng tên thư mục `.venv-app`. Sau khi kích hoạt môi trường này, cài các thư viện trong `requirements-app.txt`. File này ghim `scikit-learn==1.6.1`, phiên bản nhóm dùng để lưu các model `.joblib`.

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

Sau khi Streamlit khởi động, mở `http://localhost:8501`.

## 5. Hướng dẫn chạy tiền xử lý

### B1. Kiểm tra và làm sạch dữ liệu

Chạy các lệnh theo thứ tự để kiểm tra khả năng đọc ảnh, tìm ảnh trùng, tạo danh sách ảnh hợp lệ, rồi xuất thống kê và biểu đồ.

```bash
python scripts/build_image_manifest.py --dataset-root Sohas_weapon-Classification --output-dir outputs/data_cleaning
python scripts/find_exact_duplicates.py --manifest outputs/data_cleaning/image_manifest.csv --dataset-root Sohas_weapon-Classification --output-dir outputs/data_cleaning
python scripts/build_clean_manifest.py --manifest outputs/data_cleaning/image_manifest.csv --hashes outputs/data_cleaning/image_hashes.csv --output-dir outputs/data_cleaning
python scripts/summarize_dataset.py --manifest outputs/data_cleaning/image_manifest.csv --output outputs/data_cleaning/dataset_statistics.csv
python scripts/create_b1_figures.py --manifest outputs/data_cleaning/image_manifest.csv --dataset-root Sohas_weapon-Classification --output-dir outputs/data_cleaning --clean-manifest outputs/data_cleaning/clean_manifest.csv
```

File `outputs/data_cleaning/clean_manifest.csv` liệt kê ảnh được dùng ở bước tiếp theo. Nếu đã có file này và đang dùng đúng bộ ảnh gốc tương ứng, có thể bắt đầu từ B2.

### B2 và B3. Đổi kích thước, chia tập và trích xuất HOG

Trên Windows, bước B2 cần quyền tạo symbolic link để tạo `dataset/original` trỏ tới thư mục ảnh gốc.

```bash
python scripts/build_resized_branches.py
python scripts/create_split.py
python scripts/extract_hog_by_split.py --branch direct
python scripts/extract_hog_by_split.py --branch padding
```

Script resize tạo hai bộ ảnh `224×224`. Nhánh `direct` co giãn trực tiếp, còn nhánh `padding` giữ tỷ lệ gốc và đệm viền xám. Hai nhánh dùng chung `dataset/split.csv`, chia train/validation/test theo tỷ lệ 4/1/1. Kết quả HOG nằm trong `dataset/features/hog_direct.npz` và `dataset/features/hog_padding.npz`.

Khi cần tạo lại dữ liệu trung gian từ đầu, thêm `--clean-output` vào lệnh `build_resized_branches.py`. Cờ này xóa ảnh resize và HOG cũ, nên chạy lại cả bốn lệnh trên theo thứ tự.

### Cấu hình HOG

- Chuyển grayscale từ ảnh đã resize.
- `orientations=9`.
- `pixels_per_cell=(8, 8)`.
- `cells_per_block=(2, 2)`.
- `block_norm="L2-Hys"`.
- Ảnh `224×224` tạo 26.244 đặc trưng `float32`.

Mỗi file HOG `.npz` chứa `X`, `y`, `split`, `class_names`, `image_ids`, `relative_paths`, `sha256`, metadata duplicate và cấu hình HOG. Khi train, chỉ fit các biến đổi học được (ví dụ `StandardScaler`) trên dòng có `split == "train"`.

## 6. Huấn luyện và Đánh giá Mô hình SVM

Script `scripts/train_svm.py` thực hiện các bước sau.

- Chuẩn hóa đặc trưng bằng `StandardScaler(with_mean=False)` (chỉ fit trên `train`, áp dụng cho `val`/`test`).
- Xử lý mất cân bằng lớp (8.9:1) bằng `class_weight='balanced'`.
- Tinh chỉnh `C` cho SVM tuyến tính và `C`, `gamma` cho SVM RBF. Nhóm chọn cấu hình theo macro-F1 trên tập validation vì chỉ số này cho sáu lớp trọng số bằng nhau.
- Huấn luyện `LinearSVC` (tối ưu lồi trên 26k chiều HOG).
- Huấn luyện `SVC(RBF)` sau khi giảm chiều bằng Randomized PCA (256 chiều).
- Xuất toàn bộ bảng chỉ số (Accuracy, Balanced Accuracy, Macro Precision/Recall/F1), ma trận nhầm lẫn chuẩn hóa (Confusion Matrix heatmap) và danh sách ảnh dự đoán sai (Error Analysis).

```bash
# Chạy huấn luyện và đánh giá trên cả hai nhánh
python scripts/train_svm.py --branches direct padding
```

Toàn bộ kết quả thực nghiệm, biểu đồ và trọng số mô hình được lưu tại `outputs/svm/`.


## 7. Huấn luyện và Đánh giá Logistic Regression

Notebook `notebooks/logistic_regression.ipynb` dùng HOG nhánh `padding` (`dataset/features/hog_padding.npz`), cùng cách chia train/val/test với SVM. Notebook thực hiện các bước sau.

- Mô hình cơ sở: `StandardScaler` + `LogisticRegression(C=1.0, class_weight="balanced")` trên toàn bộ 26.244 đặc trưng.
- Lựa chọn đặc trưng bằng `SelectKBest(f_classif)` với k = 1000, 3000, 8000.
- Điều chuẩn L1 (`solver="saga"`) trên 8.000 đặc trưng đã lọc. Cell này mất khoảng 24 phút trên CPU.
- Chọn biến thể có macro-F1 cao nhất trên `val`, đánh giá trên `test` (Accuracy, Balanced Accuracy, Macro/Weighted F1, MCC, Log-loss, ROC-AUC).

Sau khi có `dataset/features/hog_padding.npz` từ mục 5, chạy notebook trong môi trường đã cài thư viện huấn luyện.

```bash
python -m pip install nbconvert ipykernel
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir notebooks --output logistic_regression_executed notebooks/logistic_regression.ipynb
```

Lệnh lưu notebook đã chạy tại `notebooks/logistic_regression_executed.ipynb`.

Kết quả được lưu tại `outputs/logistic_regression/`: bảng so sánh các biến thể trên `val`, kết quả SelectKBest, chỉ số trên `test` và xác suất dự đoán từng ảnh. Kết quả trong repo là từ lần chạy trên Google Colab (CPU). Notebook lưu model vào `lr_final.joblib` dưới dạng dict có kèm lớp `Feat` định nghĩa trong notebook, nên chỉ nạp được khi lớp này tồn tại. App dùng `lr_padding_model.joblib`: cùng pipeline `StandardScaler` + `LogisticRegression` đó, được lưu lại thành pipeline scikit-learn thuần để nạp không cần notebook.
