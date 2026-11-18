# LAB 09 — Named Entity Recognition và Information Extraction

## 31. Individual Learning Check

1. **Sequence classification khác token classification như thế nào?** Sequence classification gán một nhãn cho cả chuỗi, chẳng hạn nhãn cảm xúc cho một câu. Token classification gán nhãn tại từng vị trí cần dự đoán, chẳng hạn nhãn NER cho từng từ.
2. **BIO tagging dùng để làm gì?** BIO biểu diễn loại và ranh giới của thực thể. `B-X` mở đầu thực thể loại `X`, `I-X` tiếp tục thực thể đó và `O` đánh dấu vị trí ngoài thực thể.
3. **B-PER khác I-PER như thế nào?** Trong tên *Elon Musk*, `Elon` nhận `B-PER` vì bắt đầu tên người, còn `Musk` nhận `I-PER` vì tiếp tục cùng tên.
4. **Tại sao tokenizer gây ra label alignment problem?** Dữ liệu NER thường gán nhãn theo từ, còn tokenizer có thể tách mỗi từ thành nhiều subword và chèn token đặc biệt. Vì vậy phải ánh xạ nhãn từ sang các vị trí token mà mô hình nhận.
5. **IGNORE trong loss có ý nghĩa gì?** Nhãn `IGNORE` (thường là `-100`) đánh dấu vị trí bị loại khỏi phép tính loss. Với Strategy A, nhãn từ được gán cho subword đầu, còn subword tiếp theo, token đặc biệt và padding được bỏ qua khi tính loss.
6. **Tại sao NER không nên chỉ đánh giá bằng token accuracy?** Nhiều token mang nhãn `O`, nên mô hình đoán `O` thường xuyên vẫn có thể đạt accuracy cao dù bỏ sót nhiều thực thể. Chỉ số này cũng không cho biết mô hình lấy đủ cả cụm tên hay chưa.
7. **Entity-level F1 khác token-level F1 như thế nào?** Token-level F1 tính đúng sai tại từng vị trí. Với entity-level F1 theo exact match, một dự đoán chỉ được tính đúng khi khớp cả biên lẫn loại. Chẳng hạn, lấy mỗi *York* khi đáp án là *New York* vẫn là sai ở cấp thực thể.
8. **Vì sao entity boundary khó?** Tên thực thể có thể gồm nhiều từ, kể cả giới từ hoặc danh từ thông thường. Mô hình phải dựa vào ngữ cảnh để biết nên bắt đầu và kết thúc ở đâu. Thiếu một từ có thể làm sai cả span.
9. **Vì sao contextual representation hữu ích cho NER?** Biểu diễn của từ thay đổi theo các từ xung quanh. Chẳng hạn, *Barcelona* có thể là địa điểm trong câu về du lịch hoặc là đội bóng trong câu về kết quả thi đấu.
10. **NER có thể được sử dụng như thế nào trong Information Extraction?** NER xác định span và loại của tên người, tổ chức và địa điểm trong văn bản. Hệ thống có thể đưa các span đó vào những trường có cấu trúc hoặc dùng chúng để trích xuất quan hệ, chẳng hạn một người làm việc cho tổ chức nào.
