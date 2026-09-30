# Error Analysis — LAB 02 Language Models

## 1. Prediction đúng 1

**Context:** `the nature`  
**Model prediction:** `of`  
**Expected:** `of`  
**Probability:** `0.0002868`

**Phân tích:**  
Model dự đoán đúng vì cụm `the nature of` khá phổ biến trong dữ liệu. Đây là một pattern mà trigram có khả năng đã gặp nhiều lần trong tập train.

---

## 2. Prediction đúng 2

**Context:** `than 25`  
**Model prediction:** `years`  
**Expected:** `years`  
**Probability:** `0.0000214`

**Phân tích:**  
Model dự đoán đúng vì cụm `than 25 years` là một cách kết hợp từ tương đối tự nhiên và có khả năng xuất hiện trong corpus. Context đủ cụ thể nên trigram có thể dự đoán đúng từ tiếp theo.

---

## 3. Prediction sai 1

**Context:** `was a`  
**Model prediction:** `great`  
**Expected:** `lovely`  
**Probability:** `0.0001831`

**Phân tích:**  
Context `was a` có thể đi với rất nhiều từ như `great`, `good`, `little`, `lovely`,... Vì vậy model chọn `great` là từ có xác suất cao nhất trong train, nhưng từ thực tế trong test lại là `lovely`.

**Nguyên nhân:**  
- Context còn ngắn và có nhiều cách tiếp tục.
- Model chỉ dựa vào n-gram gần nhất.
- Tần suất trong training data ảnh hưởng mạnh đến prediction.

---

## 4. Prediction sai 2

**Context:** `in green`  
**Model prediction:** `but`  
**Expected:** `building`  
**Probability:** `0.00000642`

**Phân tích:**  
Model dự đoán `but` nhưng từ thực tế là `building`. Xác suất dự đoán cũng rất thấp, cho thấy context này không được học tốt trong tập train.

**Nguyên nhân:**  
- Training data chưa đủ cho context này.
- Có thể gặp unseen hoặc rare n-gram.
- Sparsity làm xác suất của các continuation khá thấp và gần nhau.

---

# Các lỗi khác quan sát được

## 5. Sentence Ranking tự động trả về `-inf`

Ở ví dụ sentence ranking tự động, cả ba candidate đều có:

`Log probability = -inf`

Nguyên nhân là trong candidate có ít nhất một token được model gán xác suất bằng 0. Khi đó:

`log(0) = -inf`

nên tổng log probability của cả candidate cũng bằng `-inf`.

**Nguyên nhân chính:**  
- OOV word.
- Vocabulary limitation.
- Zero probability vẫn còn xảy ra dù đã dùng Laplace.

---

## 6. Dữ liệu có ký tự nhiễu

Với context:

`Atlanta Hustle`

model dự đoán một ký tự bất thường thay vì từ thực tế `team`.

Điều này cho thấy corpus C4 vẫn có một số ký tự hoặc dữ liệu nhiễu chưa được làm sạch hoàn toàn.

**Ảnh hưởng:**  
- Làm vocabulary lớn hơn.
- Tạo ra các token ít có ý nghĩa.
- Có thể ảnh hưởng tới smoothing và next-word prediction.

---


Nhìn chung, trigram học tốt trên training data nhưng dễ gặp sparsity và unseen n-gram trên dữ liệu mới. Đây là nguyên nhân chính làm prediction và sentence ranking chưa ổn định.
