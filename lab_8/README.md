
# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

## 1. Mục tiêu
Tìm hiểu pretrained Transformer và fine-tuning BERT cho bài toán phân loại cảm xúc văn bản. So sánh hiệu năng, thời gian huấn luyện và khả năng tổng quát hóa của các mô hình.

## 2. Cấu hình
- Model: `bert-base-uncased`
- Dataset: IMDb
- Task: Binary Sentiment Classification
- Framework: PyTorch, Hugging Face Transformers
- GPU: NVIDIA L4 (Google Colab)
- Seed: 42
- Metrics: Accuracy, Precision, Recall, Macro F1

## 3. Nội dung thực nghiệm
1. Pretrained BERT vs Training from Scratch
2. Frozen Encoder vs Full Fine-Tuning
3. Training Data Efficiency (10%, 25%, 50%, 100%)
4. Maximum Sequence Length (64, 128, 256)
5. Domain Shift
6. Error Analysis và Confidence Analysis

## 4. Cấu trúc thư mục
```text
lab08/
├── README.md
├── calculations.md
├── prediction.md
├── tokenization.ipynb
├── pretrained_classification.py
├── fine_tuning.ipynb
├── results.csv
├── data_efficiency.png
├── confusion_matrix.png
├── confidence_analysis.png
├── error_analysis.md
└── reflection.md
```

## 5. Cách chạy
1. Mở `fine_tuning.ipynb` trên Google Colab.
2. Chọn GPU và cài đặt các thư viện cần thiết.
3. Chuẩn bị dataset IMDb.
4. Chạy lần lượt các cell trong notebook.
5. Kết quả được lưu trong thư mục `lab8_artifacts/`.

## 6. Kết quả chính
| Model | Accuracy | Macro F1 |
|---|---:|---:|
| From Scratch | 52.20% | 39.23% |
| Fine-Tuned BERT | 91.60% | 91.60% |
| Frozen BERT | 73.80% | 73.76% |

## 7. Kết luận
Fine-tuned BERT đạt hiệu năng cao nhất nhờ tận dụng biểu diễn ngôn ngữ từ pretraining. Frozen encoder giảm chi phí huấn luyện nhưng kém chính xác hơn. Training from scratch cần nhiều dữ liệu và thời gian huấn luyện hơn để đạt hiệu năng tốt.
