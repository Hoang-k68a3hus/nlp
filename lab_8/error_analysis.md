# 20. Error Analysis — LAB 08

**Nguồn:** `selected_errors.csv` (kết quả dự đoán thực nghiệm, không tạo ví dụ giả). **Quy ước:** `0 = Negative`, `1 = Positive`; confidence là xác suất của **nhãn model đã dự đoán**.

## 20.1. Ba false positives và ba false negatives

Cột **Text** là trích đoạn nguyên văn để bảng ngắn gọn; toàn bộ review và `source_id` nằm trong `selected_errors.csv`. Các **Hypothesis** là giả thuyết giải thích lỗi, chưa được xác minh bằng thử nghiệm riêng.

| Text | True | Pred | Confidence | Error type | Hypothesis |
|---|:---:|:---:|---:|---|---|
| **FP1** — “A creepy film; worth seeing!” | Negative (0) | Positive (1) | 99.47% | false positive; ambiguous sentiment / label ambiguity | Lời khen “worth seeing” ở cuối thiên về tích cực, trái với nhãn negative; có thể có sự không khớp giữa nhãn và sắc thái văn bản. |
| **FP2** — “The dramatic scenes can be extravagantly beautiful” … “Anyone looking for tense action is likely to be disappointed.” | Negative (0) | Positive (1) | 99.36% | false positive; ambiguous sentiment / context | Review vừa khen hình ảnh, thẩm mỹ vừa chê cảnh hành động; model có thể ưu tiên lời khen và bỏ qua đánh giá tiêu cực có điều kiện. |
| **FP3** — “didn’t surpass their original” … “this is a good family film though overall!” | Negative (0) | Positive (1) | 99.34% | false positive; negation / ambiguous sentiment / label ambiguity | Câu có so sánh phủ định nhưng phần kết khen phim; model có thể đặt trọng số lớn vào nhận xét tích cực ở cuối, trong khi nhãn là negative. |
| **FN1** — “poor cinematography” … “Most folks will want to “steer” clear.” | Positive (1) | Negative (0) | 99.77% | false negative; label ambiguity | Văn bản thể hiện nhiều nhận xét tiêu cực dù nhãn là positive; cần rà soát sự phù hợp giữa nội dung review và nhãn nguồn. |
| **FN2** — “it shows no ray of light or hope, it’s all doom & gloom” | Positive (1) | Negative (0) | 99.19% | false negative; context / label ambiguity | Các câu phê bình không khí phim mang từ ngữ tiêu cực; model có thể hiểu đó là chê phim thay vì nhận xét về một tác phẩm vẫn được gán nhãn positive. |
| **FN3** — “very, very disappointed of the acting” … “But all in all, it is an interesting movie, worth to watch it” | Positive (1) | Negative (0) | 99.13% | false negative; long-range dependency / context | Review chê nhiều khía cạnh trước khi kết luận tích cực. Model có thể bỏ sót ý kết luận hoặc bị giới hạn độ dài đầu vào; cần kiểm tra truncation thực tế. |

## 20.2. Hai trường hợp sai với confidence rất cao

| Text (review trong bảng trên) | True | Pred | Confidence | Error type | Hypothesis |
|---|:---:|:---:|---:|---|---|
| **FN1** — `test/pos/688_7.txt` | Positive (1) | Negative (0) | **99.77%** | false negative; high-confidence error | Phản hồi chủ yếu tiêu cực nhưng gán nhãn positive; cần kiểm tra khả năng nhãn không phản ánh đúng nội dung. |
| **FP1** — `train/neg/10087_4.txt` | Negative (0) | Positive (1) | **99.47%** | false positive; high-confidence error | Câu kết “worth seeing” mang sắc thái tích cực nhưng nhãn negative; có thể có sự mơ hồ về nhãn hoặc sắc thái tổng thể. |

> **Lưu ý:** CSV có 8 dòng lựa chọn, nhưng 2 dòng high-confidence lặp lại review **FN1** và **FP1**. Vì vậy có **6 review sai khác nhau** (3 FP, 3 FN), trong đó **2 review** cũng thuộc nhóm sai với confidence cao; không phải 8 review độc lập.

## 20.3. Nhận xét

- **Lỗi nổi bật:** `ambiguous sentiment`, `context`, `label ambiguity`; một số review trộn khen–chê hoặc có nhãn trái với sắc thái thể hiện trong văn bản.
- **Khả năng bỏ sót thông tin dài:** FN3 có lời khen mang tính kết luận ở cuối review; cần kiểm tra số token sau tokenization và phần bị `truncation` trước khi khẳng định nguyên nhân `long-range dependency`.
- **Confidence không đảm bảo đúng:** FN1 sai với **99.77%** và FP1 sai với **99.47%**. Cần phân biệt độ tự tin dự đoán với độ chính xác thực tế.
- **Không đủ bằng chứng** để khẳng định các nhóm `sarcasm`, `rare word`, `domain-specific expression` hoặc `tokenization` là nguyên nhân của 6 trường hợp này.

**Kết luận:** Mô hình có thể sai rất tự tin khi review chứa tín hiệu trái chiều hoặc nhãn nguồn không nhất quán với nội dung. Những nguyên nhân trên cần kiểm tra thêm bằng rà soát nhãn và kiểm tra token/truncation.
