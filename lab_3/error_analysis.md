# Error Analysis

## 1. Correct / Reasonable Cases

### Case 1 — car / automobile

**Observed:**  
Cosine similarity = **0.872**

**Expected:**  
High similarity.

**Possible explanation:**  
Hai từ gần nghĩa và xuất hiện trong context rất giống nhau.

**Evidence from corpus:**  
Cả `car` và `automobile` đều xuất hiện với `road`, `engine` và `driver`.

---

### Case 2 — cat / dog

**Observed:**  
Cosine similarity = **0.862**

**Expected:**  
High similarity.

**Possible explanation:**  
Hai từ cùng nhóm động vật và có nhiều context tương tự.

**Evidence from corpus:**  
Cả hai đều xuất hiện với `eats`, `likes` và mẫu câu `is an animal`.

---

### Case 3 — doctor / physician

**Observed:**  
Cosine similarity = **0.706**

**Expected:**  
High similarity.

**Possible explanation:**  
Hai từ gần nghĩa và thường xuất hiện trong cùng chủ đề y tế.

**Evidence from corpus:**  
Cả hai xuất hiện với `patient`, `hospital`, `disease`, `treatment` và `care`.

---

## 2. Unexpected Cases

### Case 4 — computer / banana

**Observed:**  
Cosine similarity = **0.855**

**Expected:**  
Low similarity.

**Possible explanation:**  
Corpus quá nhỏ nên model chưa học được semantic relation ổn định.

**Evidence from corpus:**  
`computer` và `banana` chỉ xuất hiện trong một số ít câu, nên vector dễ bị ảnh hưởng bởi các context chung.

---

### Case 5 — doctor / cat

**Observed:**  
`cat` xuất hiện trong Top-5 từ gần `doctor`.

Similarity = **0.809**

**Expected:**  
Low similarity.

**Possible explanation:**  
Window khá lớn và nhiều câu có cấu trúc giống nhau, làm context bị trộn.

**Evidence from corpus:**  
Cả hai thường xuất hiện trong các câu ngắn bắt đầu với `the` và có nhiều từ chức năng giống nhau.

---

### Case 6 — doctor / queen

**Observed:**  
`queen` xuất hiện trong Top-5 từ gần `doctor`.

Similarity = **0.801**

**Expected:**  
Low similarity.

**Possible explanation:**  
Corpus nhỏ, số lần xuất hiện ít và context window rộng làm model học các pattern câu thay vì nghĩa thật.

**Evidence from corpus:**  
`doctor` và `queen` đều xuất hiện trong các câu ngắn có nhiều từ phổ biến như `the`, `is`.

---

## 3. Summary

Ba trường hợp hợp lý:

- `car - automobile`: **0.872**
- `cat - dog`: **0.862**
- `doctor - physician`: **0.706**

Ba trường hợp bất ngờ:

- `computer - banana`: **0.855**
- `doctor - cat`: **0.809**
- `doctor - queen`: **0.801**

Nguyên nhân chính của các kết quả bất ngờ là corpus nhỏ, context window khá rộng và số lần xuất hiện của nhiều từ còn ít.
