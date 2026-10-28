
# LAB 06 — Attention Mechanism

Bài thực hành môn **Xử lý ngôn ngữ tự nhiên và ứng dụng**: tìm hiểu Attention, trực quan hóa trọng số chú ý và so sánh mô hình LSTM với Self-Attention trên bài toán phân loại cảm xúc IMDb.

## Cấu trúc bài nộp

```text
lab06/
├── README.md                 # Giới thiệu và hướng dẫn chạy
├── calculations.md           # Các bài tập tính attention bằng tay
├── prediction.md             # Dự đoán trước khi chạy thực nghiệm
├── attention_numpy.py        # Cài đặt attention bằng NumPy
├── self_attention.ipynb      # Các thí nghiệm, huấn luyện và đánh giá
├── results.csv               # Accuracy, F1, thời gian train, số tham số
├── attention_heatmap.png     # Heatmap trọng số self-attention
├── confusion_matrix.png     # Ma trận nhầm lẫn LSTM và Attention
├── error_analysis.md         # Phân tích dự đoán đúng/sai
└── reflection.md             # Tổng kết kiến thức và hạn chế của Attention
```

## Nội dung notebook

- **Temperature & Scaling:** ảnh hưởng đến phân phối attention.
- **Experiment 1–3:** attention distribution, thay đổi query và heatmap.
- **Experiment 4:** huấn luyện, so sánh LSTM với Self-Attention trên IMDb.
- **Experiment 5:** chi phí tính toán theo độ dài chuỗi.
- **Application & Error Analysis:** dự đoán sentiment và quan sát các lỗi.

## Hướng dẫn chạy

1. Mở `self_attention.ipynb` trên Google Colab và đặt `attention_numpy.py` cùng thư mục làm việc.
2. Tải bộ IMDb `aclImdb_v1.tar.gz` lên đường dẫn `/content/aclImdb_v1.tar.gz`.
3. Chạy **Runtime → Run all**. Notebook sẽ giải nén vào `/content/w4/aclImdb` nếu cần, sau đó gộp dữ liệu và chia Train/Validation/Test theo cách của LAB 05.
4. Các file `results.csv`, `attention_heatmap.png` và `confusion_matrix.png` được xuất từ kết quả thực nghiệm; kiểm tra lại sự nhất quán với `error_analysis.md` sau khi chạy mới.

**Lưu ý:** `calculations.md`, `prediction.md` và `reflection.md` là các phần riêng theo yêu cầu đề bài. Người nộp cần tự kiểm chứng nội dung phân tích và khai báo việc sử dụng AI theo mục 30 của LAB 06.
