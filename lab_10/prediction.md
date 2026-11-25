# 9. Prediction Before Experiment

## 9.1. Trùng từ khóa

Mô hình có thể chọn nhầm khi nhiều câu lặp từ khóa và chứa các ứng viên cùng loại. Ví dụ, hai câu cùng nói về một giải thưởng nhưng nêu hai người khác nhau. Mô hình cần xác định đúng quan hệ mà câu hỏi yêu cầu. Kiểm tra qua câu hỏi diễn đạt lại ở §22 và các ca chọn nhầm thực thể ở §24.

## 9.2. Độ dài đáp án

Đáp án dài có thể khó khớp hoàn toàn hơn vì dễ thiếu hoặc thừa token ở hai biên. EM dự kiến giảm rõ hơn F1, vì F1 vẫn ghi nhận phần token trùng. Kiểm tra theo bốn nhóm độ dài đáp án ở §23.

## 9.3. Thông tin gây nhiễu

EM và F1 dự kiến giảm khi chèn câu có từ khóa và loại đáp án tương tự câu chứng cứ. Mô hình có thể lấy span từ câu gây nhiễu. Ở §22, giữ nguyên câu hỏi và đáp án, rồi so sánh trước và sau khi chèn câu.

## 9.4. Thứ tự câu

Đảo thứ tự câu có thể làm một số dự đoán thay đổi dù câu chứa đáp án vẫn giữ nguyên. Vị trí token và thông tin lân cận thay đổi khi đảo câu. Ở §22, đo cả chênh lệch EM/F1 và số câu có dự đoán thay đổi.
