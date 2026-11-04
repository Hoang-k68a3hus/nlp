# LAB 07 — Reflection

> **Bản nháp để sinh viên tự rà soát và chỉnh sửa theo trải nghiệm thực tế trước khi nộp.**

## 1. Kiến thức rút ra

- **Self-attention** cho phép các token tương tác trực tiếp; **multi-head attention** dùng nhiều phép chiếu song song, nhưng không có nghĩa mỗi head luôn học một quy tắc ngôn ngữ riêng.
- **Positional encoding** cung cấp thông tin thứ tự mà self-attention thuần túy không tự có.
- **Residual connection** tạo đường truyền thông tin trực tiếp; **LayerNorm** hỗ trợ ổn định quá trình huấn luyện; **FFN** biến đổi phi tuyến biểu diễn của từng token.

## 2. Liên hệ với kết quả thực nghiệm

Notebook ghi nhận rằng tăng sequence length từ 32 lên 256 giúp cải thiện F1 macro trung bình từ khoảng 0,7242 lên 0,8703, nhưng attention memory tăng mạnh. Điều này cho thấy cần cân nhắc giữa lượng ngữ cảnh giữ lại và chi phí tính toán. Việc tăng số attention heads không bảo đảm chất lượng tăng đều. So sánh LSTM và Transformer cũng không cho phép kết luận một kiến trúc luôn tốt hơn trong mọi điều kiện.

## 3. Hạn chế và điều cần kiểm chứng

- Cắt chuỗi ở `max_length` có thể làm mất phần chứa tín hiệu cảm xúc quan trọng.
- Những câu có phủ định, mỉa mai, đối lập ý hoặc cảm xúc mơ hồ vẫn khó phân loại.
- Attention heatmap chỉ biểu diễn trọng số attention, **không phải bằng chứng nhân quả** giải thích quyết định của mô hình.
- Kết luận về hiệu năng cần gắn với tập dữ liệu, seed, cấu hình và thời gian chạy cụ thể.

## 4. Kết nối với LAB 08

LAB 07 tập trung vào Transformer Encoder được xây dựng và huấn luyện cho bài toán cụ thể. LAB 08 chuyển sang mô hình Transformer đã pretrain như BERT, sau đó fine-tune. Nền tảng về attention, positional information, residual, LayerNorm và FFN giúp hiểu kiến trúc này.

## 5. Tự đánh giá (sinh viên hoàn thiện)

- **Thành phần tôi có thể tự giải thích và triển khai:** [Tự điền]
- **Khó khăn thực tế khi làm LAB 07 và cách tôi khắc phục:** [Tự điền]
- **Điều tôi muốn kiểm chứng hoặc cải thiện ở lần chạy tiếp theo:** [Tự điền]
