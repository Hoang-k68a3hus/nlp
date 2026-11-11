# LAB 08 — Prediction Before Experiment

*Các dự đoán dưới đây được ghi trước khi chạy thí nghiệm, chưa phải kết quả thực tế.*

## Prediction 1 — Pretrained vs. From Scratch

- **Prediction:** BERT pretrained sau fine-tuning sẽ đạt Accuracy và F1 cao hơn model train from scratch trên cùng dữ liệu IMDb.
- **Reason:** Pretraining đã giúp encoder học biểu diễn ngôn ngữ từ kho văn bản lớn; model from scratch phải học biểu diễn và phân loại từ đầu.
- **Expected evidence:** Với cùng tập dữ liệu và điều kiện đánh giá, Accuracy/F1 trên validation của pretrained cao hơn, loss thường giảm nhanh hơn.

## Prediction 2 — Frozen Encoder vs. Full Fine-tuning

- **Prediction:** **C. Kém hơn** full fine-tuning.
- **Reason:** Frozen encoder giữ nguyên biểu diễn tổng quát, chỉ classification head được học. Full fine-tuning còn điều chỉnh encoder theo đặc trưng đánh giá cảm xúc của IMDb.
- **Expected evidence:** Full fine-tuning có Accuracy/F1 validation cao hơn frozen; mức chênh lệch có thể nhỏ tùy dữ liệu và cấu hình.

## Prediction 3 — Chỉ dùng 10% Training Data

- **Prediction:** Model **train from scratch** sẽ giảm performance mạnh hơn pretrained model khi giảm từ 100% xuống 10% training data.
- **Reason:** Pretrained model đã có biểu diễn ngôn ngữ học từ trước nên cần ít dữ liệu hơn để thích nghi với tác vụ; scratch phụ thuộc nhiều vào dữ liệu huấn luyện để học cả biểu diễn lẫn phân loại.
- **Expected evidence:** So với mức 100%, Accuracy/F1 của scratch giảm nhiều hơn; kiểm chứng bằng cách huấn luyện và đánh giá cả hai model trên các mức dữ liệu tương ứng.
