# Reflection — LAB 06: Attention Mechanism

> **Bản nháp tham khảo:** Sinh viên cần tự đối chiếu với kết quả thí nghiệm, chỉnh sửa theo hiểu biết của mình và khai báo hỗ trợ AI theo mục 30 của đề.

## 1. Bài học chính

Attention tạo **context vector** bằng cách dùng Query so sánh với Key, chuẩn hóa scores bằng softmax rồi tính tổng có trọng số của Value. Scaling giúp hạn chế scores quá lớn; temperature điều chỉnh mức độ tập trung của attention weights.

## 2. LSTM và Self-Attention

LSTM dùng **final hidden state** sau quá trình xử lý tuần tự. Self-Attention cho phép token tương tác trực tiếp với các token khác và tổng hợp **contextual representations**. Trong kết quả đã lưu, Attention đạt **85,26% Accuracy**, cao hơn LSTM (**82,44%**), nhưng điều này không có nghĩa Attention luôn tốt hơn.

## 3. Những giới hạn cần lưu ý

Heatmap chỉ cho thấy **phân bố attention**, không chứng minh mô hình hiểu ngôn ngữ hoặc giải thích được nguyên nhân dự đoán. Full self-attention có ma trận kích thước **T × T**, nên chi phí tăng theo **O(T²)**. Self-attention cơ bản không có positional encoding cũng chưa thể biểu diễn thứ tự token một cách tường minh.

## 4. Kết nối với LAB 07

LAB 06 giúp làm rõ cơ chế Q/K/V và context representation, làm nền tảng để tìm hiểu **Multi-Head Attention, Positional Encoding và Transformer** ở LAB 07.
