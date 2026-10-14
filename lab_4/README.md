
# LAB 04 – Text Classification

## 1. Giới thiệu
Bài thực hành phân loại cảm xúc đánh giá phim trên bộ dữ liệu IMDb bằng các phương pháp biểu diễn văn bản và mô hình học máy.

## 2. Dataset
- Dataset: IMDb Movie Reviews
- Tổng số: 50.000 reviews
- Nhãn: Negative (0), Positive (1)
- Chia dữ liệu: Train 70%, Validation 15%, Test 15%

## 3. Phương pháp
- TF-IDF (Unigram, Unigram + Bigram)
- Average Word2Vec
- Logistic Regression
- Linear SVM

Các phương pháp biểu diễn chỉ được fit trên tập train để tránh data leakage.

## 4. Cấu trúc bài nộp
- `classification.ipynb`: Thí nghiệm và đánh giá mô hình.
- `preprocessing.py`: Xử lý văn bản và mô hình baseline.
- `calculations.tex`: Bài tập tính toán.
- `prediction.tex`: Dự đoán trước thực nghiệm.
- `results.csv`: Kết quả đánh giá.
- `confusion_matrix.png`: Ma trận nhầm lẫn.
- `error_analysis.md`: Phân tích các trường hợp dự đoán đúng và sai.
- `reflection.tex`: Tổng kết bài thực hành.

## 7. Kết luận
Bài thực hành giúp hiểu ảnh hưởng của cách biểu diễn văn bản đến hiệu quả phân loại, đồng thời nhận biết hạn chế của TF-IDF và Average Word2Vec trong xử lý ngữ cảnh.
