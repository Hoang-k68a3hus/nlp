# LAB 05 — 24. Error analysis

Phân tích 6 review trong `error_analysis.csv` của mô hình GRU: 3 dự đoán đúng và 3 dự đoán sai. Cả 6 review đều có nhãn thật là **Negative**; 3 lỗi là **False Positive** (dự đoán Positive thay vì Negative).

| STT | Text (trích đoạn) | True | Predicted | Analysis |
|---|---|---|---|---|
| 1 | “mediocre at best”, “as funny as a bag of hammers” | Negative | Negative | **Đúng.** Có lời chê rõ và cách nói mỉa mai. Chưa biết mô hình có hiểu mỉa mai hay chỉ dựa vào từ tiêu cực. |
| 2 | “characters are unlikeable”, “script is awful” | Negative | Negative | **Đúng.** Từ ngữ tiêu cực khá trực tiếp, dễ nhận diện. |
| 3 | “It is hilarious ... I cannot see anything positive in this film” | Negative | Negative | **Đúng.** Có ý khen xen phủ định; mô hình vẫn dự đoán Negative. CSV ghi 5 token `UNK`. |
| 4 | “Although ... was excellent, this story is too painful to watch” | Negative | Positive | **Sai.** Có thể mô hình chú ý vào `excellent` mà chưa xử lý tốt ý đối lập phía sau. CSV ghi 1 token `UNK`. |
| 5 | “beginning did have elements of style ... but the plot ... completely loses it” | Negative | Positive | **Sai.** Có cả khen và chê; review bị cắt **51 token** và có 6 token `UNK`. Chưa biết phần bị cắt có làm đổi dự đoán không. |
| 6 | “they do a pretty good job ... what is their problem when it comes to the movies?” | Negative | Positive | **Sai.** Đầu review có lời khen, nhưng toàn review mang nhãn Negative. Bị cắt **838 token**, có 6 token `UNK`; mô hình có thể đã bỏ lỡ ý chê ở phần sau. |

## Trả lời — Lỗi có phải do sequence modeling không?

**Chưa thể khẳng định.** Mẫu 4 và 5 có ý khen–chê đối lập nên lỗi *có thể* liên quan đến việc kết hợp ngữ cảnh theo thứ tự. Mẫu 5 và đặc biệt mẫu 6 còn bị **truncation**, tức mất thông tin từ bước xử lý đầu vào, không phải bằng chứng trực tiếp về vanishing gradient. Các token `UNK` cũng có thể ảnh hưởng, nhưng chưa xác định được mức độ.

**Kết luận:** Mô hình dự đoán đúng các trường hợp có tín hiệu tiêu cực tương đối rõ, nhưng nhầm ở ba review có ngữ cảnh pha trộn hoặc bị cắt. Không thể kết luận đơn giản rằng “GRU không hiểu câu”. Muốn biết nguyên nhân chính cần đọc toàn văn và thử lại với `max_length` lớn hơn hoặc thay đổi từng cụm từ quan trọng.