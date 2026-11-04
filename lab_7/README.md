
# LAB 07 — Transformer Architecture

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** From Attention to Transformer  
**Dataset:** IMDb Movie Reviews  
**Framework:** PyTorch

## 1. Giới thiệu

LAB 07 tập trung xây dựng Transformer Encoder từ các thành phần cơ bản và ứng dụng vào bài toán phân loại cảm xúc văn bản (Sentiment Classification).

Các thành phần chính:

- Multi-Head Self-Attention
- Positional Encoding
- Residual Connection và Layer Normalization
- Feed-Forward Network (FFN)
- Transformer Encoder
- Pooling và Classification

## 2. Cấu trúc thư mục

```text
lab07/
├── README.md
├── calculations.md
├── prediction.md
├── transformer_block.py
├── positional_encoding.py
├── transformer_classification.ipynb
├── results.csv
├── positional_encoding.png
├── attention_heads.png
├── learning_curves.png
├── confusion_matrix.png
├── error_analysis.md
└── reflection.md
```

## 3. Dataset

Sử dụng IMDb Movie Reviews gồm 50.000 đánh giá phim, chia thành hai nhãn:

- `0`: Negative
- `1`: Positive

Dữ liệu được đọc từ `aclImdb_v1.tar.gz`.

Quy trình xử lý:

1. Đọc và tổng hợp dữ liệu IMDb.
2. Chia train/validation/test theo tỷ lệ xấp xỉ 70/15/15, nhóm các văn bản trùng nhau.
3. Tokenization và xây dựng vocabulary chỉ trên tập train.
4. Padding, truncation và chuyển văn bản thành tensor.
5. Đưa dữ liệu vào mô hình.

## 4. Kiến trúc mô hình

```text
Input Text
    |
    v
Tokenization
    |
    v
Embedding
    |
    v
Positional Encoding
    |
    v
Multi-Head Self-Attention
    |
    v
Residual + LayerNorm
    |
    v
Feed-Forward Network
    |
    v
Residual + LayerNorm
    |
    v
Pooling
    |
    v
Linear Classifier
    |
    v
Positive / Negative
```

## 5. Các thí nghiệm

| Experiment | Nội dung |
|---|---|
| 1 | Positional Encoding và Permutation |
| 2 | Number of Attention Heads |
| 3 | Attention Head Diversity |
| 4 | Transformer vs LSTM |
| 5 | Sequence Length |
| 6 | Pooling Strategy |

Các kết quả, biểu đồ và nhận xét được trình bày trong `transformer_classification.ipynb`.

## 6. Kết quả chính

- Mô hình Transformer được xây dựng từ các thành phần PyTorch cơ bản.
- Positional Encoding cung cấp thông tin thứ tự token.
- Số lượng attention heads ảnh hưởng đến kết quả nhưng không đảm bảo cải thiện F1.
- Tăng sequence length giúp giữ lại nhiều ngữ cảnh hơn nhưng làm tăng chi phí tính toán.
- LSTM và Transformer được so sánh trên cùng bài toán phân loại cảm xúc.
- Mean Pooling và CLS-style Pooling được đánh giá thực nghiệm.

### Kết quả Transformer

Checkpoint: `transformer_h4_l128_mean_pe1_s42`

| Metric | Value |
|---|---:|
| Test samples | 7,502 |
| Correct predictions | 6,265 |
| Incorrect predictions | 1,237 |
| Accuracy | 83.51% |

## 7. Hướng dẫn chạy

### Yêu cầu

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook hoặc Google Colab

### Cài đặt thư viện

```bash
pip install torch numpy pandas matplotlib scikit-learn jupyter
```

### Chuẩn bị dữ liệu

Đặt file `aclImdb_v1.tar.gz` trong thư mục `/content/` nếu chạy trên Google Colab.

### Chạy notebook

Mở:

```text
transformer_classification.ipynb
```

Chọn:

```text
Runtime → Run all
```

Notebook sẽ thực hiện các bước xử lý dữ liệu, huấn luyện, đánh giá và xuất kết quả.

## 8. Error Analysis

Phân tích các trường hợp dự đoán sai với những yếu tố:

- Negation
- Contrast
- Long sequence
- Truncation
- Rare words
- Ambiguous sentiment
- Domain-specific terms
- Noise

Attention visualization được sử dụng để quan sát phân phối attention weights, không được xem là bằng chứng nhân quả giải thích dự đoán.

Chi tiết trong `error_analysis.md`.

## 9. Kết luận

LAB 07 minh họa cách kết hợp Self-Attention, Positional Encoding, FFN, Residual Connection và LayerNorm để tạo thành Transformer Encoder.

Các thí nghiệm cho thấy hiệu quả của Transformer phụ thuộc vào cấu hình kiến trúc, độ dài chuỗi và cách tổng hợp biểu diễn. Đồng thời, mô hình có những hạn chế về chi phí attention và khả năng xử lý văn bản dài.

LAB 07 là nền tảng để tiếp cận Pretrained Transformer và BERT trong LAB 08.
