# LAB 10 — Error Analysis

Các ví dụ lấy từ kết quả BERT-Tiny fine-tuned, max length 384, epoch 3 trên 1.200 câu test của SQuAD v1.1. Phần §22 dùng thêm bộ 12 cặp tự soạn. Offset là khoảng ký tự gốc nửa mở `[start, end)`.

## 20. Experiment 3 — Năm ca EM = 0 và F1 > 0

Trong 970 câu có EM = 0, 344 câu vẫn có F1 > 0. F1 theo bảng dùng thang 0–1. Công thức và chuẩn hóa nằm trong [calculations.md](calculations.md).

| Question | Gold | Prediction | EM | F1 | Vì sao F1 > 0? |
| --- | --- | --- | --- | --- | --- |
| What coalition rose up to oppose Louis XIV's France? | League of Augsburg | Augsburg | 0 | 0,500 | Giữ “Augsburg”, bỏ “League of”. Span mất phần xác định đây là tên liên minh. |
| On what date did the Florida legislature decide against the plan to renovate the Miami stadium? | May 3, 2013 | 3, 2013 | 0 | 0,800 | Giữ “3” và “2013”, thiếu tháng “May”. Câu hỏi yêu cầu ngày đầy đủ. |
| Who did Noble acquire the rights to the American Broadcasting Company name from? | George B. Storer | George B. Storer in 1944 | 0 | 0,750 | Giữ đủ ba token tên người và thêm “in 1944”. Câu hỏi chỉ hỏi người chuyển nhượng. |
| Who did Genghis Khan assign as his successor? | Ögedei Khan | Ögedei | 0 | 0,667 | Giữ tên “Ögedei” nhưng bỏ “Khan”, một phần của gold. |
| What type of living does cycling promote? | healthy | healthy living… The authority | 0 | 0,500 | Giữ tính từ “healthy” và kéo span sang phần còn lại của câu cùng đầu câu kế tiếp. |

## 22. Experiment 4 — Question/Context Matching

| Điều kiện | n | EM (%) | F1 (%) | Số prediction đổi so với gốc |
| --- | --- | --- | --- | --- |
| Từ khóa trực tiếp | 12 | 50,00 | 68,92 | 0 |
| Câu hỏi diễn đạt lại | 12 | 41,67 | 54,68 | 8 |
| Chèn câu gây nhiễu | 12 | 8,33 | 24,69 | 10 |
| Đảo thứ tự câu | 12 | 25,00 | 57,96 | 6 |

- **Prediction 1.** Tỷ lệ phủ từ nội dung qua câu chứng cứ giảm từ 83,73% xuống 0% khi diễn đạt lại. EM giảm 8,33 điểm và F1 giảm 14,25 điểm. Riêng câu về Calder Medal, prediction trực tiếp là “Maya Chen”, còn biến thể diễn đạt lại lấy một cụm dài chứa cả tên và năm. Sự khác biệt có thể đến từ cả lựa chọn từ lẫn mức độ trực tiếp của câu hỏi.
- **Prediction 3.** Khi chèn một câu với thực thể tương tự, 10/12 prediction thay đổi. Với câu Calder Medal, model trả “Sara Bell”, người nhận Porter Medal ở câu gây nhiễu. Điều này minh họa cách match từ khóa có thể dẫn đến nhầm quan hệ.
- **Prediction 4.** Đảo hai câu làm đổi 6/12 prediction. Ở câu hỏi về bài báo tháng 6, cụm đầu ra “Sofia Martin” bị mở rộng thành “recycled paper. Sofia Martin” sau khi câu mô tả giấy in được đặt lên trước. Đáp án và câu chứa đáp án vẫn giữ nguyên.
- Đây là phép thử chẩn đoán trên 12 cặp. EM có thể đổi 8,33 điểm khi chỉ một câu thay đổi trạng thái khớp, nên cần mẫu lớn hơn để ước lượng mức chênh đáng tin cậy.

## 23. Experiment 5 — Answer Length

| Gold dài (token) | n | EM (%) | F1 (%) |
| --- | --- | --- | --- |
| 1 | 425 | 25,41 | 31,25 |
| 2–3 | 536 | 18,28 | 31,13 |
| 4–6 | 163 | 11,04 | 30,23 |
| 7+ | 76 | 7,89 | 24,85 |

**Prediction 2 phù hợp với kết quả:** EM giảm từ 25,41% ở nhóm 1 token xuống 7,89% ở nhóm 7+ token. F1 giữ quanh 30–31% trong ba nhóm đầu và giảm xuống 24,85% ở nhóm dài nhất. Khi đáp án dài, mô hình còn có thể lấy được một phần đúng dù lệch biên. Nhóm 7+ chỉ có 76 câu và câu hỏi giữa các nhóm có thể khác về độ khó, nên đây là quan sát trên bài chạy.

## 24. Error Analysis — 15 ca đọc lại từ context

### 24.1. Năm ca wrong start

| Question | Gold | Prediction | Error type | Gold → predicted offset | Giải thích |
| --- | --- | --- | --- | --- | --- |
| In what state is the largest Huguenot Society located? | Texas | Manakin Episcopal Church in Virginia as an historic shrine with occasional services. The Society has chapters in numerous states, with the one in Texas | Wrong start · câu gây nhiễu | [715, 720) → [569, 720) | Giữ biên cuối “Texas” nhưng bắt đầu từ tên một nhà thờ ở Virginia trong câu trước. “Texas” mới là bang có chi hội lớn nhất. |
| What is a chemical secreted by tumors that suppresses the immune response? | cytokine TGF-β | β | Wrong start · một phần thuật ngữ | [302, 316) → [315, 316) | Giữ ký tự cuối của “TGF-β”. Span phải bắt đầu từ “cytokine” để lấy đủ tên chất; phần β riêng lẻ không định danh chất. |
| What late Gothic architectural sculptural feature is found on the tower above the main entrance? | an open work crown surmounted by a statue of fame | pier is Romanesque in form but Classical in detail. Likewise the tower above the main entrance has an open work crown surmounted by a statue of fame | Wrong start · sai biên câu | [603, 652) → [504, 652) | Giữ toàn bộ mô tả chiếc vương miện ở cuối span nhưng lấy cả đoạn mô tả Romanesque đứng trước câu chứa đáp án. |
| What is an example of a problem to which effective algorithms have provided a solution in spite of the intractability associated with the breadth of sizes? | NP-complete knapsack | knapsack | Wrong start · đáp án thiếu thành phần | [399, 419) → [411, 419) | Chọn “knapsack”, bỏ định ngữ “NP-complete” chỉ biến thể của bài toán được nêu trong câu. |
| What future Revolutionary key figures participated in this attack? | Washington and Thomas Gage | Thomas Gage | Wrong start · bỏ sót thực thể | [530, 556) → [545, 556) | Chọn người đứng cuối danh sách. Câu hỏi dùng số nhiều, và chứng cứ nêu cả Washington lẫn Thomas Gage. |

### 24.2. Năm ca wrong end

| Question | Gold | Prediction | Error type | Gold → predicted offset | Giải thích |
| --- | --- | --- | --- | --- | --- |
| What nationality is Hoesung Lee? | Korean | Korean economist Hoesung Lee is the chair of the IPCC since October 8, 2015 | Wrong end · sai biên cụm | [0, 6) → [0, 75) | Bắt đầu đúng ở “Korean” nhưng lấy luôn nghề, tên, chức vụ và ngày. Câu hỏi chỉ yêu cầu quốc tịch. |
| What was the Chinese name for the Central Secretariat? | Zhongshu Sheng | Zhongshu Sheng) at Khanbaliq | Wrong end · sai biên ngoặc | [287, 301) → [287, 315) | Bắt đầu đúng ở “Zhongshu Sheng” nhưng đi qua dấu ngoặc đóng và lấy thêm nơi đặt cơ quan. |
| How much did the population of Victoria increase in ten years after the discovery of gold? | sevenfold | sevenfold from 76,000 | Wrong end · thông tin thừa | [544, 553) → [544, 565) | Chọn đúng hệ số tăng “sevenfold” và lấy thêm dân số mốc ban đầu “from 76,000”. |
| What app did viewers use to watch the game on their smartphones?  | NFL Mobile service | NFL Mobile service. The ESPN Deportes Spanish broadcast was made available through WatchESPN | Wrong end · câu gây nhiễu | [304, 322) → [304, 396) | Chọn đúng tên ứng dụng ở đầu span rồi nối sang câu nói về WatchESPN. Câu thứ hai kể về một kênh phát khác. |
| Name one country that banned boating, driving and flying on Sundays. | Italy | Italy, Switzerland | Wrong end · nhiều câu trả lời khả dĩ | [334, 339) → [334, 352) | Văn bản kể nhiều nước, còn câu hỏi yêu cầu một nước. Span bắt đầu đúng ở Italy nhưng thêm Switzerland thành danh sách hai nước. |

### 24.3. Năm ca wrong span

| Question | Gold | Prediction | Error type | Gold → predicted offset | Giải thích |
| --- | --- | --- | --- | --- | --- |
| Which part of the V&A collection did the Salting Bequest enhanced? | Chinese and Japanese ceramics | 1909 | Wrong span · sai loại đáp án | [785, 814) → [742, 746) | “1909” là năm lập di tặng. Phần bộ sưu tập được bổ sung nằm sau mệnh đề mô tả tác động của di tặng. |
| Which year resulted in the suspension of one of the two soccer teams? | 2014 | 2005 | Wrong span · nhầm mốc cùng loại | [13, 17) → [5, 9) | Cả 2005 và 2014 xuất hiện trong cùng khoảng thời gian. Mô hình lấy mốc bắt đầu, trong khi biến cố đình chỉ gắn với mốc sau. |
| At what university's facility did the Broncos practice? | Stanford | Jose State practice facility and stayed at the San Jose Marriott | Wrong span · nhầm chủ thể | [117, 125) → [26, 90) | Đoạn trích nói về sân tập và nơi ở của Panthers. Stanford xuất hiện ở câu kế tiếp nói riêng về Broncos. |
| What kind of non-peer-reviewed sources does the IPCC use? | model results | grey literature | Wrong span · mức khái quát của câu hỏi | [439, 452) → [328, 343) | “Grey literature” gọi chung nhóm nguồn không qua bình duyệt. Gold chỉ một ví dụ cụ thể được liệt kê ở câu sau. |
| How much is terra preta distributed over the Amazon forest? | large areas | 2003 by Michael Heckenberger and colleagues of the University of Florida | Wrong span · lệch chứng cứ trong context dài | [53, 64) → [569, 641) | Prediction lấy năm và tên nhóm nghiên cứu ở một đoạn cách xa đáp án. “Large areas” nằm cạnh từ “distributed” ngay phần đầu. |

Các nhãn trên mô tả điều quan sát được trong văn bản và offset. Ca TGF-β liên quan đến biên thuật ngữ, nhưng đầu ra riêng lẻ chưa cho biết subword là nguyên nhân. Truncation được chứng minh bằng ca có kiểm soát ở §16. Ở thí nghiệm §14, cả 1.200 câu test đều có ít nhất một gold thấy trọn trong một cửa sổ tại cả ba độ dài, nên các ca trên không được quy cho việc mọi cửa sổ đều mất gold. Câu cố ý thiếu bằng chứng được chạy riêng ở §27.

## 25. Wrong Start vs Wrong End

Đối chiếu với một gold tham chiếu được chọn theo thứ tự EM, F1, số biên trùng, rồi khoảng cách giữa các biên. Thống kê sau dùng chỉ số ký tự và tính cả trường hợp dự đoán có token trùng một phần.

| Phạm vi | n | Chỉ sai start | Chỉ sai end | Sai hai biên | Tổng sai start | Tổng sai end |
| --- | --- | --- | --- | --- | --- | --- |
| Chỉ câu có EM = 0 | 970 | 114 | 104 | 752 | 866 | 856 |
| Toàn bộ test | 1.200 | 117 | 106 | 755 | 872 | 861 |

Trong 970 lỗi EM, 752 ca lệch hai biên. Cả việc chọn sai vùng chứng cứ lẫn việc lấy cụm quá rộng đều có thể tạo lỗi hai biên. Sai start nhỉnh hơn sai end 10 câu. Ví dụ cho thấy sai start có thể bỏ định ngữ hoặc lấy từ câu trước, còn sai end thường thể hiện ở việc bỏ chức danh hoặc kéo thêm ngày, số liệu và câu kế tiếp. EM vẫn có thể bằng 1 dù lệch offset nếu normalization bỏ phần dư hoặc cùng đáp án xuất hiện ở vị trí khác.
