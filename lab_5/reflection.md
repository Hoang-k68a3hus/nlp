# LAB 05 — Reflection

Thí nghiệm trên IMDb cho thấy mô hình chuỗi có thể xử lý thứ tự từ thông qua hidden state, khác với TF-IDF unigram. Trong lần chạy đã lưu, Test F1 của RNN là **0.7425**, LSTM là **0.8140** và GRU là **0.8408**. Tuy vậy, TF-IDF + Logistic Regression vẫn đạt **0.8903**, nên mô hình phức tạp hơn không chắc cho kết quả tốt hơn.

Khi tăng `max_length` từ 50 lên 200, F1 của GRU tăng nhưng chi phí tính toán cũng tăng. Ở thí nghiệm đổi vị trí từ `excellent` và `terrible`, xác suất dự đoán thay đổi nhưng cả sáu câu đều được dự đoán Positive. Điều này cho thấy mô hình còn hạn chế; thí nghiệm không đủ để chứng minh vanishing gradient.

RNN phải truyền thông tin qua nhiều bước liên tiếp, nên việc học quan hệ xa có thể khó khăn. LSTM và GRU dùng các gate để hỗ trợ giữ thông tin, nhưng vẫn xử lý tuần tự. **Attention** là hướng tiếp theo vì có thể giúp mô hình truy cập trực tiếp những token quan trọng ở vị trí xa.
