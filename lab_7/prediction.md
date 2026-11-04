# LAB 07 — Prediction

> Các dự đoán dưới đây được ghi **trước khi chạy thực nghiệm**. Phần kiểm chứng ở cuối file chỉ được điền sau khi có kết quả thực tế; không sửa dự đoán ban đầu.

## Prediction 1

**Question:** Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?

- **Prediction:** Có.
- **Reason:** RNN cập nhật hidden state theo từng timestep, nên bước sau phụ thuộc bước trước. Self-attention trong Transformer có thể tính tương tác giữa các token đồng thời trong một tầng, thuận lợi hơn cho xử lý song song.
- **Confidence:** 95%.

## Prediction 2

**Question:** Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?

- **Prediction:** Không, nếu không có bất kỳ nguồn thông tin vị trí nào khác.
- **Reason:** Self-attention thuần túy dựa trên nội dung token và không tự gắn nhãn vị trí đầu, giữa hay cuối câu. Vì vậy, cần thêm positional information để biểu diễn thứ tự.
- **Confidence:** 95%.

## Prediction 3

**Question:** Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?

- **Prediction:** Không thể phân biệt dựa vào **thứ tự** khi sử dụng self-attention không có thông tin vị trí và một phép pooling không phụ thuộc thứ tự (chẳng hạn mean pooling).
- **Reason:** Hoán vị token chỉ hoán vị tương ứng các vector đầu ra của self-attention. Khi mean pooling, hai chuỗi có cùng các token sẽ tạo ra cùng biểu diễn gộp, nên bộ phân loại không nhận biết được khác biệt thứ tự.
- **Confidence:** 90%.

## Prediction 4

**Question:** Hai attention heads có nhất thiết học cùng một attention pattern không?

- **Prediction:** Không.
- **Reason:** Mỗi head có bộ ma trận chiếu Q, K, V riêng, nên có thể học các phân phối attention khác nhau. Tuy nhiên, các head cũng có thể học pattern tương tự nhau; không thể khẳng định sự đa dạng nếu chưa quan sát attention matrices.
- **Confidence:** 95%.

## Prediction 5

**Question:** Tăng số attention heads có chắc chắn làm accuracy tăng không?

- **Prediction:** Không.
- **Reason:** Với `d_model` cố định, tăng số heads làm giảm số chiều mỗi head. Việc có nhiều heads không đảm bảo các heads học được thông tin bổ sung hữu ích; accuracy còn phụ thuộc dữ liệu, cách huấn luyện và các siêu tham số.
- **Confidence:** 95%.

## Prediction 6

**Question:** Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

- **Prediction:** Không.
- **Reason:** Transformer có lợi thế xử lý song song nhưng self-attention thông thường cần tính ma trận tương tác có kích thước `T × T`. Thời gian thực tế phụ thuộc độ dài chuỗi, kích thước mô hình, batch size, phần cứng và cách triển khai; vì vậy không thể khẳng định luôn nhanh hơn RNN.
- **Confidence:** 90%.

## Prediction verification — Sau thực nghiệm

> Theo yêu cầu LAB 07, giữ nguyên các dự đoán phía trên. Chỉ ghi kết quả từ thực nghiệm thực tế và giải thích rõ nếu dự đoán sai.

| # | Dự đoán ban đầu | Kết quả thực nghiệm | Đúng/Sai | Giải thích dựa trên bằng chứng |
|---|---|---|---|---|
| 1 | Có | Chưa chạy | Chưa xác định | Điền sau khi đối chiếu cách xử lý và thời gian chạy. |
| 2 | Không | Chưa chạy | Chưa xác định | Điền sau thí nghiệm bỏ positional encoding. |
| 3 | Không phân biệt dựa trên thứ tự khi pooling bất biến theo hoán vị | Chưa chạy | Chưa xác định | So sánh biểu diễn `dog bites man` và `man bites dog`. |
| 4 | Không nhất thiết | Chưa chạy | Chưa xác định | So sánh attention matrices giữa các heads. |
| 5 | Không chắc chắn | Chưa chạy | Chưa xác định | Đối chiếu Accuracy/F1 khi `n_heads = 1, 2, 4, 8`. |
| 6 | Không phải luôn luôn | Chưa chạy | Chưa xác định | So sánh thời gian LSTM và Transformer theo sequence length. |
