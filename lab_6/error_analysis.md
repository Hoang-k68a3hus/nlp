# Error Analysis — LAB 06

Phân tích **6 mẫu** từ `error_analysis.csv`: **3 dự đoán đúng** và **3 dự đoán sai**. Cả 6 mẫu đều có nhãn thực tế **Negative (0)**.

| Index | Trích đoạn review | Nhãn thật | Dự đoán | P(Positive) | Kết quả |
|---:|---|---|---|---:|---|
| 1 | This film is mediocre at best. Angie Harmon is as funny as a bag of hammers. Her… | Negative | Negative | 4,15% | Đúng |
| 2 | The characters are unlikeable and the script is awful. It's a waste of the talents… | Negative | Negative | <0,01% | Đúng |
| 3 | The action in this movie beats Sunny bhai in Gadar. Akshay Kumar possess the… | Negative | Negative | 12,30% | Đúng |
| 0 | Sorry everyone,,, I know this is supposed to be an "art" film,, but wow, they should… | Negative | Positive | 51,22% | Sai |
| 10 | I figured that it's about time I let this one out. Pokémon fans are suffering in… | Negative | Positive | 97,44% | Sai |
| 12 | When I was 11, Grease 2 was like crack. It was a classless, shameful, euphoric, and… | Negative | Positive | 73,13% | Sai |

## Nhận xét ngắn

- **Ba mẫu đúng (1, 2, 3):** Model dự đoán Negative chính xác. Mẫu 3 có dấu hiệu *negation* và token ngoài vocabulary (UNK), nhưng vẫn được phân loại đúng.
- **Mẫu sai 0:** Model dự đoán Positive với xác suất **51,22%**, khá sát ngưỡng 50%. Dữ liệu ghi nhận dấu hiệu phủ định, tương phản và UNK.
- **Mẫu sai 10:** Model dự đoán Positive với **97,44%**. Dữ liệu ghi nhận văn bản bị cắt **838 tokens**, kèm phủ định, tương phản và UNK.
- **Mẫu sai 12:** Model dự đoán Positive với **73,13%**. Dữ liệu ghi nhận văn bản bị cắt **344 tokens**, kèm phủ định, tương phản và UNK.

**Kết luận:** Truncation, negation, contrast và UNK là các **giả thuyết cần kiểm tra**, không phải nguyên nhân lỗi đã được chứng minh. File CSV chỉ chứa đoạn review rút gọn và không có attention weights, nên chưa thể kết luận model tập trung vào token nào hoặc vì sao dự đoán sai.
