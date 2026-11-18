## 12. Prediction Before Experiment

**P1 — Transformer so với baseline.** Dự đoán Transformer pretrained đạt entity-level F1 cao hơn baseline dựa trên đặc trưng từ vì nó dùng ngữ cảnh hai chiều để phân biệt các tên dễ nhầm loại và dùng WordPiece để biểu diễn từ chưa gặp. So sánh F1 của hai mô hình trên cùng tập test.

**P2 — Độ dài thực thể.** Dự đoán thực thể dài như *United Nations Security Council* khó nhận diện đúng toàn bộ hơn tên một từ như *Microsoft*. Với exact match, thiếu hoặc thừa một token cũng làm cả thực thể bị tính sai. Kiểm tra F1 theo các nhóm 1, 2, 3 và 4+ token.

**P3 — Tần suất thực thể.** Dự đoán thực thể hiếm dễ bị bỏ sót hơn vì mô hình học từ ít ngữ cảnh có nhãn. So sánh recall theo số lần cặp (tên thực thể, loại) xuất hiện trong tập train, chia thành ba nhóm 0, 1–2 và từ 3 lần trở lên.
