# SOHAS Weapon Classification

Project đã chuẩn bị dữ liệu SOHAS theo luồng: Dataset Collection + Exploration
-> kiểm tra/làm sạch ảnh -> tạo hai nhánh resize -> chia train/validation/test -> trích xuất HOG.
(2 nhánh resize - 1 nhánh resize ảnh direct, 1 nhánh resize ảnh có padding (tùy model))
## Cấu trúc thư mục

```text
Sohas_weapon-Classification/   # ảnh raw theo 6 thư mục lớp, không chỉnh sửa
outputs/b1/                    # metadata, thống kê và cleaning của B1
scripts/                       # mã nguồn chạy pipeline
dataset/                       # ảnh resize, split và HOG sinh ra khi chạy B2/B3
```

## Chức năng từng thư mục

- `Sohas_weapon-Classification/`: dataset gốc gồm 6 lớp `billete`, `knife`, `monedero`, `pistol`, `smartphone`, `tarjeta`.
- `outputs/b1/clean_manifest.csv` là nguồn quyết định ảnh nào được phép dùng tiếp (`is_in_clean_dataset=True`).
- `scripts/`: các script CLI; không sửa raw data.
- `dataset/original/`: symlink tới dataset gốc, không phải bản copy.
- `dataset/resized/direct/`: ảnh `224×224` resize trực tiếp, có manifest và log lỗi riêng.
- `dataset/resized/padding/`: ảnh giữ aspect ratio, padding xám trung tính `BGR=(128,128,128)`, có manifest và log lỗi riêng.
- `dataset/split.csv`: mapping từng ảnh clean sang `train`, `val` hoặc `test`. Đã lọc các ảnh trùng SHA256.
- `dataset/features/`: HOG cho hai nhánh.

## Chức năng từng script

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

## 3. Yêu cầu hệ thống và cài đặt

Khuyến nghị Python 3.10 trở lên; project đã được kiểm tra với Python 3.14.4. Nên dùng virtual environment để tránh xung đột thư viện.

```bash
# Tạo môi trường ảo
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# Cài dependency
python -m pip install -r requirements.txt
```

Dataset SOHAS phải nằm tại `Sohas_weapon-Classification/`. Nếu chưa có kết quả B1, cần chạy các script B1 để tạo `outputs/data_cleaning/clean_manifest.csv` trước.

## 4. Hướng dẫn chạy tiền xử lý

Chạy các lệnh sau từ thư mục root của project.

```bash
# Bước 1: Tạo hai nhánh ảnh 224×224.
# --clean-output xóa các output tạo dở/cũ của pipeline trước khi chạy lại.
python scripts/build_resized_branches.py --clean-output

# Bước 2: Chia tập train/validation/test theo tỷ lệ 4/1/1.
python scripts/create_split.py

# Bước 3: Trích xuất HOG cho nhánh direct resize.
python scripts/extract_hog_by_split.py --branch direct

# Bước 4: Trích xuất HOG cho nhánh padding xám.
python scripts/extract_hog_by_split.py --branch padding
```

`--branch` là cờ duy nhất cần chọn: `direct` cho ảnh bị ép trực tiếp thành hình vuông, hoặc `padding` cho ảnh giữ aspect ratio với padding xám trung tính.

## Cấu hình HOG

- Chuyển grayscale từ ảnh đã resize.
- `orientations=9`.
- `pixels_per_cell=(8, 8)`.
- `cells_per_block=(2, 2)`.
- `block_norm="L2-Hys"`.
- Ảnh `224×224` tạo 26.244 đặc trưng `float32`.

Mỗi file HOG `.npz` chứa `X`, `y`, `split`, `class_names`, `image_ids`, `relative_paths`, `sha256`, metadata duplicate và cấu hình HOG. Khi train, chỉ fit các biến đổi học được (ví dụ `StandardScaler`) trên dòng có `split == "train"`.
