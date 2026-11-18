# LAB 09 · Phân tích dự đoán NER

Phân tích dùng 3.453 câu CoNLL-2003 test chuẩn. Test ID là chỉ số câu bắt đầu từ 1 sau khi bỏ hàng `-DOCSTART-`. Checkpoint được chọn theo validation entity F1. Tokenizer WordPiece uncased, Strategy A ở mục 10.

## 21. Experiment 4 — Nhãn thực thể

| entity_type | train_entities | support | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| PER | 6600 | 1617 | 82.49% | 86.21% | 84.31% |
| ORG | 6321 | 1661 | 67.99% | 68.27% | 68.13% |
| LOC | 7140 | 1668 | 82.90% | 86.03% | 84.44% |
| MISC | 3438 | 702 | 63.13% | 66.10% | 64.58% |

PER và LOC dẫn đầu trong lần chạy đi kèm. Tín hiệu họ tên nhiều từ và cụm chỉ nơi chốn hỗ trợ hai nhãn này. ORG vẫn có 6.321 mẫu train, nhưng tên đội có thể trùng địa danh và tên tổ chức có thể chứa những từ thông dụng. MISC có ít mẫu nhất và bao gồm nhiều dạng như quốc tịch, tên giải. Sự khác biệt giữa văn xuôi và bảng tỉ số còn thể hiện ở việc một surface được dùng cho cả thành phố và đội bóng. Các yếu tố này là diễn giải từ ví dụ, chưa được kiểm định nhân quả.

## 22. Experiment 5 — Độ dài thực thể

| entity_length | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| 1 | 3574 | 72.09% | 79.94% | 75.81% |
| 2 | 1776 | 86.49% | 80.01% | 83.12% |
| 3 | 236 | 70.81% | 55.51% | 62.23% |
| 4+ | 62 | 54.55% | 29.03% | 37.89% |

F1 của nhóm hai từ cao hơn nhóm một từ. Giảm mạnh ở nhóm ba từ và 4+ ủng hộ một phần dự đoán về lỗi biên của cụm dài. Chỉ 62 gold span thuộc nhóm 4+, nên kết quả nhóm này nhạy với từng lỗi. Độ dài được tính theo từ CoNLL, và mỗi predicted span được xếp nhóm bằng độ dài của chính nó.

## 23. Experiment 6 — Lỗi thực tế

Transformer có **1,397 FP** và **1,221 FN** ở mức span. Bảng chọn 5 FP không chồng gold, 5 FN không chồng dự đoán, 5 lỗi sai biên cùng type và 2 lỗi sai type đúng biên. Gold/Prediction là thực thể đang phân tích trong câu. Mỗi lỗi sai biên và sai loại cũng tạo một FP và một FN theo exact-match. Giải thích dùng từ ngữ, context, WordPiece và tần suất train quan sát được.

| Test ID | Text | Gold | Prediction | Error type | Explanation |
| --- | --- | --- | --- | --- | --- |
| 45 | Bitar saved well again from Miura in the 37th minute , parrying away his header from a corner . | O ở vùng dự đoán | parrying [PER] | FP tách rời | Động từ phân từ chỉ động tác cản bóng. Câu có nhiều họ cầu thủ, nhưng từ này nằm sau dấu phẩy và theo sau bằng away. WordPiece vùng phân tích: `parry ##ing`. |
| 107 | Saqlain Mushtaq b Harris 2 | O ở vùng dự đoán | b [PER] | FP tách rời | Ký hiệu b trong dòng điểm cricket chen giữa hai tên cầu thủ. Dạng dòng thiếu câu hoàn chỉnh làm dấu hiệu ngữ pháp yếu. WordPiece vùng phân tích: `b`. |
| 153 | Bowyer , 19 , who was caught in the act by security cameras , pleaded guilty to a charge of affray at a court in London . | O ở vùng dự đoán | affray [LOC] | FP tách rời | Trong charge of affray, từ chỉ hành vi bị truy tố. Cấu trúc at a court in London ở sau mới cung cấp địa điểm. WordPiece vùng phân tích: `af ##fra ##y`. |
| 193 | Centre Jason Little will miss Australia 's end-of-tour fixture against the Barbarians at Twickenham on Saturday . | O ở vùng dự đoán | miss [ORG] | FP tách rời | Từ đi ngay sau will, nên là động từ. Jason Little và Australia hai bên tạo các tên riêng lân cận có thể gây nhiễu. WordPiece vùng phân tích: `miss`. |
| 493 | W L T PF PA | O ở vùng dự đoán | PA [ORG] | FP tách rời | PA nằm cùng W, L, T và PF trong hàng tiêu đề thống kê. Gold xem đây là một ký hiệu cột. WordPiece vùng phân tích: `pa`. |
| 16 | Bitar pulled off fine saves whenever they did . | Bitar [PER] | O ở vùng gold | FN tách rời | Chủ ngữ họ đơn lẻ đi với pulled off fine saves; không có tên đệm hoặc danh xưng. Gold surface/type chưa xuất hiện trong train. WordPiece vùng phân tích: `bit ##ar`. Số lần surface/type trong train: 0. |
| 143 | SOCCER - BLINKER BAN LIFTED . | BLINKER [PER] | O ở vùng gold | FN tách rời | Tiêu đề SOCCER viết hoa toàn dòng, nên chữ hoa không giúp tách họ cầu thủ khỏi các từ khác. Bản uncased còn bỏ thông tin hoa/thường. WordPiece vùng phân tích: `blink ##er`. Số lần surface/type trong train: 0. |
| 194 | Little has opted not to risk aggravating the knee injury which ruled him out of a large chunk of the tour and is replaced by fellow Queenslander Daniel Herbert . | Little [PER] | O ở vùng gold | FN tách rời | Little có dạng giống tính từ tiếng Anh. Cụm has opted và ngữ cảnh chấn thương ủng hộ cách đọc đây là họ người. WordPiece vùng phân tích: `little`. Số lần surface/type trong train: 1. |
| 219 | League title-holders Steaua Bucharest , who finished bottom of their Champions ' League group in the European Cup , have only two players in the squad . | Champions ' League [MISC] | O ở vùng gold | FN tách rời | Tên giải có dấu nháy tách thành một từ gốc riêng. Model bỏ cả cụm MISC trong ngữ cảnh group; dấu nháy là trở ngại khả dĩ. WordPiece vùng phân tích: `champions / ' / league`. Số lần surface/type trong train: 0. |
| 561 | Pace , a junior , helped Ohio State to a 10-1 record and a berth in the Rose Bowl against Arizona State . | Rose Bowl [MISC] | O ở vùng gold | FN tách rời | Cụm a berth in the Rose Bowl chỉ giải/trận đấu. Surface/type này vắng trong train và hai thành phần đều là từ thông dụng. WordPiece vùng phân tích: `rose / bowl`. Số lần surface/type trong train: 0. |
| 8 | The former Soviet republic was playing in an Asian Cup finals tie for the first time . | Soviet [MISC] | Soviet republic [MISC] | Sai biên | Model giữ đúng MISC nhưng ghép thêm danh từ chung republic sau tính từ Soviet. WordPiece vùng phân tích: `soviet`. Số lần surface/type trong train: 9. |
| 29 | Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages . | 1995 World Cup [MISC] | World Cup [MISC] | Sai biên | Model nhận đúng lõi World Cup, bỏ năm 1995 mà gold gộp vào tên kỳ giải. WordPiece vùng phân tích: `1995 / world / cup`. Số lần surface/type trong train: 0. |
| 117 | B. Young c Moin Khan b Waqar 5 | B. Young [PER] | Young [PER] | Sai biên | Model bỏ chữ cái viết tắt B. ở đầu tên. B. là một từ CoNLL nhưng có nhiều mảnh WordPiece. WordPiece vùng phân tích: `b . / young`. Số lần surface/type trong train: 0. |
| 145 | Dutch forward Reggie Blinker had his indefinite suspension lifted by FIFA on Friday and was set to make his Sheffield Wednesday comeback against Liverpool on Saturday . | Sheffield Wednesday [ORG] | Sheffield [ORG] | Sai biên | Wednesday thường là tên ngày. Trong trường hợp này nó là phần bắt buộc của tên câu lạc bộ. WordPiece vùng phân tích: `sheffield / wednesday`. Số lần surface/type trong train: 4. |
| 170 | Teamsystem Bologna ( Italy ) 9 7 2 16 | Teamsystem Bologna [ORG] | Bologna [ORG] | Sai biên | Hàng bảng điểm thiếu động từ; model giữ phần địa danh và bỏ Teamsystem là phần tên đội đã tách subword. WordPiece vùng phân tích: `teams ##yst ##em / bologna`. Số lần surface/type trong train: 0. |
| 20 | Japan , co-hosts of the World Cup in 2002 and ranked 20th in the world by FIFA , are favourites to regain their title here . | FIFA [ORG] | FIFA [MISC] | Sai loại | Giữ đúng biên nhưng nhầm ORG sang MISC. Cụm ranked … by FIFA mô tả cơ quan xếp hạng, là dấu hiệu cho ORG. WordPiece vùng phân tích: `fifa`. Số lần surface/type trong train: 1. |
| 26 | on Friday for their friendly against Scotland at Murrayfield more than a year after the 30-year-old wing announced he was retiring following differences over selection . | Murrayfield [LOC] | Murrayfield [PER] | Sai loại | Model giữ biên nhưng gán PER. Giới từ at trong ngữ cảnh đá giao hữu là dấu hiệu tên sân/địa điểm. WordPiece vùng phân tích: `murray ##field`. Số lần surface/type trong train: 0. |

### Kiểm tra Prediction 3 · Tần suất trong train

| train_count | gold | tp_exact | recall |
| --- | --- | --- | --- |
| 0 | 2618 | 1796 | 68.60% |
| 1–2 | 570 | 391 | 68.60% |
| 3+ | 2460 | 2240 | 91.06% |

Nhóm có từ ba lần xuất hiện có recall cao hơn nhóm vắng hoặc hiếm trong train. Nhóm 0 và 1–2 gần bằng nhau. Tần suất là số lần cặp (type, surface gộp chữ hoa/thường) xuất hiện trong CoNLL train, nên chưa phản ánh dữ liệu pretraining hay tác động riêng của độ dài và type.

## 24. Context Analysis — Cùng surface, khác loại gold

| Surface | Test ID | Text quanh thực thể | Gold | Baseline | Transformer |
| --- | --- | --- | --- | --- | --- |
| Barcelona | 147 | … 's players ' status committee , meeting in ⟦Barcelona⟧ , decided that although the Udinese document was … | LOC | Barcelona [LOC] | Barcelona [ORG] |
| Barcelona | 180 | ⟦Barcelona⟧ ( Spain ) 9 4 5 13 | ORG | Barcelona [ORG] | Barcelona [ORG] |
| Charlton | 3093 | ⟦Charlton⟧ 2 Swindon 0 | ORG | Charlton [ORG] | Charlton [ORG] |
| Charlton | 3446 | ⟦Charlton⟧ , 61 , and his wife , Peggy … | PER | Charlton [PER] | Charlton [PER] |
| Hartford | 410 | ⟦Hartford⟧ 4 BOSTON 2 | ORG | Hartford [ORG] | Hartford [ORG] |
| Hartford | 1631 | BALANCE - ⟦Hartford⟧ , Conn . | LOC | Hartford [PER] | Hartford [LOC] |
| Toronto | 2319 | " The negotiations will be held both in ⟦Toronto⟧ and in Jakarta , " one source , … | LOC | Toronto [LOC] | Toronto [LOC] |
| Toronto | 3374 | NY RANGERS 6 ⟦Toronto⟧ 5 | ORG | Toronto [ORG] | Toronto [ORG] |

- **Barcelona.** meeting in đánh dấu thành phố; tìm tên đội ở đầu dòng bảng điểm có (Spain) và dãy điểm. BERT nhầm câu 147 sang ORG trong khi baseline nhận đúng LOC.

- **Charlton.** Tỉ số giữa Charlton và Swindon chỉ đội; tuổi và his wife trong câu 3446 chỉ người. Xét lần nhắc đầu tiên ở câu 3446.

- **Hartford.** Hàng tỉ số với Boston gợi đội; dấu phẩy và viết tắt bang Conn. gợi thành phố. BERT dự đoán đúng LOC ở 1631, baseline gán PER.

- **Toronto.** held both in … and in … gợi nơi đàm phán; tỉ số cạnh NY Rangers gợi đội thể thao.

Embedding tĩnh của một surface dùng cùng một vector khởi đầu ở mọi ngữ cảnh. Self-attention đưa thông tin từ các từ còn lại trong câu vào hidden state tại vị trí đang xét, sau đó token head dự đoán BIO. Dòng điểm, giới từ chỉ nơi chốn và thông tin tuổi/gia đình vì thế có thể làm thay đổi dự đoán. Baseline ở đây cũng có đặc trưng ±1 từ và có thể xử lý một số ca đổi loại. Barcelona ở câu 147 cho thấy Transformer vẫn có thể nhầm.
