LAB 01 — From Text Processing to Search

Mục tiêu

Lab này xây dựng và đánh giá một hệ thống tìm kiếm văn bản dựa trên TF-IDF và cosine similarity.

Cấu trúc bài nộp

lab01/
├── README.md
├── calculations.md
├── prediction.md
├── implementation.py
├── experiments.ipynb
├── results.csv
└── reflection.md

Nội dung các file

calculations.md: bài tính tay về Count Vector, TF, IDF, TF-IDF và cosine similarity.

prediction.md: các dự đoán trước khi chạy experiment trên corpus 30K.

implementation.py: tự cài đặt các hàm TF-IDF và cosine similarity, unit test và so sánh với thư viện tham chiếu.

experiments.ipynb: các experiment trên corpus 30K, gồm sparse representation, preprocessing ablation, document search, evaluation và error analysis.

results.csv: kết quả retrieval và evaluation của các pipeline.

reflection.md: tổng kết kết quả, failure case, hạn chế và hướng cải tiến.

Cách chạy

Cài các thư viện cần thiết:

pip install numpy pandas scikit-learn transformers

Chạy phần core implementation:

python implementation.py

Sau đó mở experiments.ipynb, đặt file corpus:

c4-train.00000-of-01024-30K.json.gz

đúng đường dẫn được khai báo trong notebook và chạy toàn bộ các cell.

Các phần chính của experiment

Xây TF-IDF sparse matrix trên khoảng 30.000 documents.

Kiểm tra vocabulary, sparsity, document frequency, IDF và TF-IDF.

So sánh ba preprocessing pipelines.

Xây document search bằng cosine similarity.

Đánh giá bằng Precision@5, Recall@5 và MRR.

Phân tích các query tốt, query kém và failure case.

Đưa ra hypothesis cho semantic representation tốt hơn TF-IDF.

Ghi chú

TF-IDF được sử dụng làm lexical baseline. Kết quả cho thấy preprocessing ảnh hưởng đáng kể đến vocabulary, sparsity và chất lượng retrieval. Hệ thống vẫn có hạn chế khi query và document gần nghĩa nhưng không có lexical overlap phù hợp.
