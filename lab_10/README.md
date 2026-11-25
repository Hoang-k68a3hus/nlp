# LAB 10 — Extractive Question Answering

Môn **Xử lý ngôn ngữ tự nhiên và ứng dụng**, Học kỳ I – 2026. Chủ đề **From Token Classification to Span Prediction**.

## 1–5. Bài toán và đầu ra

Trong extractive QA, hệ thống nhận câu hỏi cùng văn bản chứa câu trả lời và chọn hai vị trí token `start ≤ end`. Hai offset tương ứng xác định phần văn bản gốc được trích. SQuAD v1.1 cung cấp `context`, `question`, một hoặc nhiều `answer.text` và `answer_start` tính theo ký tự. Ví dụ, với câu hỏi “What is the capital of France?” và context “The capital of France is Paris.”, đáp án là `context[25:30]`.

| Mục 5 | Biểu diễn dùng ở head | Đầu ra |
|---|---|---|
| LAB 08 — Sequence Classification | Vector `[CLS]` | Một nhãn cho cả văn bản |
| LAB 09 — NER | Các vector token | Một nhãn BIO trên mỗi vị trí được giám sát |
| LAB 10 — Extractive QA | Các vector token của cặp câu hỏi và context | Hai biên token cho một span |

## 6–28, 30–34. Đối chiếu đề và file

| Mục đề | File | Nội dung |
|---|---|---|
| §§6–8, 12, 18–19 | [calculations.md](calculations.md) | Tính start/end, loss theo log tự nhiên, offset ký tự, EM và F1 |
| §9 | [prediction.md](prediction.md) | Bốn dự đoán trước khi chạy |
| §§10–13, 15–16 | [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb) | Kiểm tra input BERT, offset, sliding window và các trường hợp biên |
| §§4–5, 8, 13–21, 26 | [qa_model.py](qa_model.py) | Encoder BERT, head QA, loss, bộ giải mã, EM/F1 và baseline |
| §§14, 17–25, 27 | [qa_finetuning.ipynb](qa_finetuning.ipynb) | Fine-tune ba cấu hình, đánh giá, kiểm tra có kiểm soát, thống kê lỗi, chạy ứng dụng và xuất kết quả |
| §§14, 17, 20, 22–23, 25 | [results.csv](results.csv) | Số đo, có cột `section` để lọc |
| §17 | [learning_curves.png](learning_curves.png) | Loss train/validation và EM/F1 validation theo epoch |
| §20 | [em_f1_analysis.png](em_f1_analysis.png) | So sánh baseline và BERT-Tiny trên cùng test |
| §23 | [answer_length_analysis.png](answer_length_analysis.png) | EM/F1 theo độ dài gold và số mẫu từng nhóm |
| §§20, 22–25 | [error_analysis.md](error_analysis.md) | Ca EM = 0/F1 > 0, đối chiếu dự đoán, 15 ca lỗi và thống kê biên |
| §27 | [document_qa.py](document_qa.py) | CLI đọc tài liệu tiếng Anh do người dùng cung cấp và trích answer span |
| §§26, 28, 30–33, 35–36 | [README.md](README.md) | Mở rộng no-answer, QA/RAG, tái lập, checklist và câu hỏi cá nhân |
| §34 | [reflection.md](reflection.md) | Sơ đồ backbone/head và bài phản ánh 300–500 từ |

PDF chuyển từ §28 sang §30. Các file giữ nguyên số mục như trong đề.

## 14, 17, 20, 22–23. Kết quả thí nghiệm

### 14. Context Length

| Max length | Cửa sổ train | Cửa sổ test | EM (%) | F1 (%) | Train (giây) |
| --- | --- | --- | --- | --- | --- |
| 128 | 10.273 | 3.066 | 18,25 | 29,20 | 141,0 |
| 256 | 4.645 | 1.337 | 19,33 | 30,01 | 129,8 |
| 384 | 4.248 | 1.211 | 19,17 | 30,65 | 177,7 |

**Tăng context length có luôn cải thiện QA không?** Không. 256 đạt EM cao nhất. 384 đạt F1 cao nhất nhưng có EM thấp hơn 256 và thời gian train dài hơn. Cả ba cấu hình đều giữ trọn ít nhất một gold cho từng câu hỏi test sau khi gộp cửa sổ. Cấu hình 128 sinh nhiều cửa sổ train, dẫn đến nhiều lần cập nhật hơn với cùng batch size và số epoch.

### 17. Pretrained QA Model

Kết quả theo epoch với cấu hình 384, đây là mô hình dùng cho các mục sau.

| Epoch | Train loss | Validation loss | Validation EM (%) | Validation F1 (%) |
| --- | --- | --- | --- | --- |
| 1 | 4,202 | 3,944 | 4,33 | 12,49 |
| 2 | 3,568 | 3,587 | 14,33 | 24,36 |
| 3 | 2,984 | 3,487 | 17,17 | 27,07 |

Checkpoint epoch 3 đạt F1 validation cao nhất. Trên 1.200 câu test, loss theo cửa sổ là 3,339. [Learning curves](learning_curves.png).

### 20. EM vs F1 và 21. Baseline

| Model | EM (%) | F1 (%) |
| --- | --- | --- |
| Keyword baseline | 18,33 | 22,72 |
| BERT-Tiny fine-tuned | 19,17 | 30,65 |

Trên cùng tập test, BERT-Tiny cao hơn 0,83 điểm EM và 7,93 điểm F1. Chênh lệch rõ hơn ở F1 cho thấy model thường lấy được nhiều token đúng hơn. [Biểu đồ EM/F1](em_f1_analysis.png) · [Các ca khớp một phần](error_analysis.md#20-experiment-3--năm-ca-em--0-và-f1--0).

### 22. Question/Context Matching

Trên 12 cặp, EM/F1 từ 50,00/68,92 ở câu hỏi trực tiếp xuống 41,67/54,68 khi diễn đạt lại, 8,33/24,69 khi chèn câu gây nhiễu và 25,00/57,96 khi đảo câu. [Phân tích theo cặp](error_analysis.md#22-experiment-4--questioncontext-matching).

### 23. Answer Length

EM giảm từ 25,41% ở nhóm 1 token xuống 7,89% ở nhóm 7+ token. Chênh lệch F1 hẹp hơn, phù hợp với việc span dài dễ giữ được một phần câu trả lời. [Biểu đồ độ dài](answer_length_analysis.png) · [Số liệu theo nhóm](error_analysis.md#23-experiment-5--answer-length).

## 26. No-Answer Extension

SQuAD v1.1 cung cấp câu hỏi có đáp án. Nhãn `[CLS]` trong bài dùng cho **cửa sổ** không chứa trọn đáp án, dù cửa sổ khác của tài liệu có thể chứa đáp án. Với SQuAD 2.0, câu hỏi thực sự không có đáp án cần nhãn `[CLS]` trên mọi cửa sổ. Sau khi train, hiệu chỉnh ngưỡng chênh điểm span và `[CLS]` trên validation của SQuAD 2.0. `predict_examples` có tham số `null_threshold` cho phép áp dụng ngưỡng đã hiệu chỉnh. Kết quả nộp dùng SQuAD v1.1, còn phần mở rộng §26 được trình bày ở mức thiết kế.

## 27. Application — Document QA

Chạy các lệnh này từ thư mục `lab10`. CLI dùng checkpoint 384 do notebook tạo nếu có, hoặc dùng thư mục `checkpoint_384` đi kèm gói ZIP. Có thể chạy demo ngay sau khi cài thư viện.

```bash
python document_qa.py --demo
python document_qa.py --document path/to/document.txt --question "What is the refund policy?"
```

Dùng `--question` nhiều lần để hỏi cùng tài liệu hoặc `--model` để chỉ định checkpoint khác. Tài liệu là UTF-8. CLI trả JSON từng dòng với đáp án, vị trí ký tự đầu, biên phải exclusive, số cửa sổ và chênh logit so với `[CLS]`. Chênh logit là điểm xếp hạng, cần hiệu chỉnh nếu muốn dùng như độ tin cậy.

| Câu hỏi demo | Kết quả mong đợi | Span model thực sự trả về |
| --- | --- | --- |
| Where is the Northbridge Library located? | Boston | Boston. It opens at 9 am on weekdays. Library cards are valid for two years. The main reading room closes at 6 pm |
| When does the Northbridge Library open on weekdays? | 9 am | 9 am |
| How long are library cards valid? | two years | two years |
| Who designed the Northbridge Library? | Không có trong tài liệu | Northbridge Library is located in Boston. It opens at 9 am |

Mô hình lấy đúng giờ mở cửa và thời hạn thẻ. Câu hỏi địa điểm chứa “Boston” trong span nhưng model đặt biên cuối quá xa. Câu hỏi về người thiết kế thiếu chứng cứ ngay trong tài liệu và mô hình vẫn trích một span có sẵn, minh họa nhu cầu hiệu chỉnh cơ chế từ chối ở §26.

## 28. Phân biệt QA và RAG

Extractive QA trong LAB 10 nhận ngữ cảnh được cung cấp sẵn và chọn một chuỗi nguyên văn trong đó. RAG tiếp nhận câu hỏi, dùng retriever tìm tài liệu liên quan, rồi đưa các đoạn tìm được vào mô hình trả lời. Một hệ thống có thể dùng reader trích xuất sau retrieval, hoặc dùng generator để tổng hợp theo cách tiếp cận trong LAB 12. Retriever quyết định tài liệu nào được đưa tới mô hình trả lời.

## 30. Reproducibility

| Thành phần | Cấu hình |
|---|---|
| Model | BERT-Tiny `google/bert_uncased_L-2_H-128_A-2`, revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. Head hai logit khởi tạo mới |
| Tokenizer | `BertTokenizerFast`, WordPiece uncased từ cùng checkpoint |
| Dataset | SQuAD v1.1 tiếng Anh |
| Train | 4.200 câu từ official train; chọn ngẫu nhiên không hoàn lại với seed 42 |
| Validation | 600 câu từ 24 bài viết nhóm đầu của official dev sau khi sắp tên rồi xáo với seed 42 |
| Test | 1.200 câu từ 24 bài viết nhóm sau của official dev, tách bài viết với validation |
| Split selection | Mỗi tập lấy mẫu không hoàn lại với cùng seed. Test chỉ dùng để báo cáo, chọn epoch theo F1 validation |
| Max sequence length | 128, 256 và 384. Cấu hình chính 384 |
| Stride | 64 trên thí nghiệm kiểm soát. 128 trong minh họa §15 |
| Giới hạn câu hỏi / đáp án | 48 WordPiece cho câu hỏi, tối đa 30 token cho span dự đoán |
| Batch size / padding | 32 feature mỗi batch, padding theo batch |
| Learning rate / epochs | `2e-4` cố định, 3 epoch cho mỗi độ dài |
| Optimizer | AdamW, weight decay 0,01, gradient clipping 1,0 |
| Random seed / checkpoint | 42. Chọn F1 validation lớn nhất. Nếu hòa, chọn epoch sớm hơn |
| Hardware | AMD EPYC 9V74, CPU, PyTorch 6 luồng |
| Môi trường đã chạy | Python 3.12, PyTorch 2.8.0 CPU, Transformers 4.51.3, pandas 2.2.3, NumPy 2.3.5, Matplotlib 3.10.8 |
| Training time | 128: 141,0 giây · 256: 129,8 giây · 384: 177,7 giây |

Cài môi trường Python 3.10 trở lên, chạy từ thư mục `lab10`.

```bash
python -m pip install "torch==2.8.0" "transformers==4.51.3" "pandas>=2.2,<3" "numpy<3" "matplotlib>=3.8,<4" "jupyterlab>=4,<5"
jupyter lab
```

1. Chạy toàn bộ `tokenizer_alignment.ipynb`.
2. Chạy toàn bộ `qa_finetuning.ipynb`. Notebook tải SQuAD v1.1 và checkpoint gốc ở lần đầu, rồi xuất CSV và ba PNG trong thư mục hiện tại.
3. Chạy `python document_qa.py --demo` với checkpoint 384 do notebook tạo.

Mặc định là hồ sơ `quick` dùng số mẫu trong bảng. Đặt biến môi trường `LAB10_PROFILE=full` **trước khi mở Jupyter** nếu muốn dùng toàn bộ official train và toàn bộ 48 bài viết official dev chia đôi theo bài viết. Hồ sơ `full` xuất tệp riêng vào `.lab10/full_results`. CPU/GPU và phiên bản thư viện có thể làm sai khác điểm và thời gian khi chạy lại. Mô hình dùng để đánh giá Document QA lưu tại `.lab10/runs/384` ở hồ sơ `quick`.

Nguồn dữ liệu và checkpoint: [SQuAD Explorer](https://rajpurkar.github.io/SQuAD-explorer/) · [Google BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f).

## 31–32. Deliverables và Rubric

Toàn bộ 13 file được liệt kê ở bảng đối chiếu. Theory & calculation và prediction nằm trong các Markdown tương ứng. Hai notebook chứa đầu ra đã chạy, gồm bảng alignment, kết quả fine-tune và đánh giá. Phân tích lỗi đọc các offset và câu chứng cứ từ lượt chấm test. Document QA có lệnh chạy độc lập với notebook sau khi có checkpoint.

## 33. Individual Learning Check

1. **Extractive QA khác classification như thế nào?** Classification gán một nhãn từ tập nhãn cố định. Extractive QA trả hai vị trí và trích một đoạn thay đổi theo ngữ cảnh.
2. **Tại sao QA cần start và end position?** Start xác định điểm bắt đầu, end xác định điểm kết thúc. Cùng một start có thể tạo các đáp án lồng nhau với độ dài khác nhau.
3. **Character offset khác token index như thế nào?** Offset đếm ký tự trong chuỗi nguồn. Token index đếm đơn vị đầu vào của mô hình, gồm subword và token đặc biệt.
4. **Tại sao cần offset mapping?** Nó nối nhãn ký tự của dataset với vị trí token để huấn luyện, đồng thời cho phép trích câu trả lời nguyên văn sau suy luận.
5. **Stride dùng để làm gì?** Nó giữ một phần context chồng lấp giữa hai cửa sổ liên tiếp, giúp span sát điểm cắt có cơ hội nằm trọn trong cửa sổ sau.
6. **Vì sao context dài cần nhiều window?** Encoder giới hạn số token trên mỗi đầu vào, gồm cả câu hỏi và token đặc biệt. Chia cửa sổ giữ được phần văn bản nằm ngoài giới hạn của cặp đầu tiên.
7. **Exact Match đo điều gì?** Tỷ lệ câu hỏi mà prediction giống hoàn toàn ít nhất một gold sau chuẩn hóa.
8. **F1 của QA khác gì với accuracy?** F1 dùng precision và recall trên token của đáp án, nên cho điểm từng phần. Accuracy chấm tỷ lệ quyết định hoàn toàn đúng theo đơn vị đánh giá đã chọn.
9. **Vì sao có thể F1 > 0 nhưng EM = 0?** Gold “New York City” và prediction “New York” có hai token chung, nhưng hai chuỗi chuẩn hóa khác nhau.
10. **Wrong-start và wrong-end khác nhau thế nào?** Wrong-start lệch ký tự đầu, như “York City” so với “New York City”. Wrong-end lệch biên cuối, như “New York” so với cùng gold.
11. **Extractive QA khác RAG ở đâu?** Bài QA này đã biết context và trích span. RAG tìm context từ một tập tài liệu rồi mới trả lời, thường bằng mô hình sinh.
12. **Khi nào extractive QA không thể trả lời?** Khi tài liệu thiếu đáp án hoặc đáp án cần suy luận/tổng hợp mà không có đoạn liên tục tương ứng trong văn bản. Với cửa sổ hữu hạn, mô hình cũng có thể bỏ lỡ một đáp án bị chia ở biên.

## 35–36. Bridge to LAB 11 và Mosaic LAB 08–12

LAB 10 đọc ngữ cảnh đã biết. LAB 11 dùng embedding để truy xuất các tài liệu liên quan từ kho lớn. LAB 12 đưa kết quả truy xuất vào mô hình sinh câu trả lời. Chuỗi bài chuyển đơn vị dự đoán từ nhãn văn bản, sang nhãn token, sang biên span, rồi đến tài liệu truy xuất và câu trả lời có căn cứ.
