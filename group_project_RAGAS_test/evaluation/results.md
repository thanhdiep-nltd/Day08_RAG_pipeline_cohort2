# RAG Evaluation Results

## Framework sử dụng

> Ghi rõ framework đã chọn: RAGAS (Version 0.4.3)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.753 | 0.560 | +0.193 |
| Answer Relevance | 0.679 | 0.508 | +0.172 |
| Context Recall | 0.750 | 0.770 | -0.020 |
| Context Precision | 0.829 | 0.675 | +0.154 |
| **Average** | 0.753 | 0.628 | +0.125 |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Jina Reranking):**
- Sử dụng Hybrid Search gộp Dense và Lexical (BM25) search.
- Có áp dụng mô hình Reranking Jina Cross-Encoder v2 (`jina-reranker-v2-base-multilingual`).
- Số lượng tài liệu đưa vào context: top_k = 5.

**Config B (Hybrid Search, Không Reranking):**
- Sử dụng Hybrid Search gộp Dense và Lexical (BM25) search.
- KHÔNG áp dụng Reranking, sắp xếp candidates theo điểm truy xuất thuần túy.
- Số lượng tài liệu đưa vào context: top_k = 5.

**Kết luận:**
- Config A (có reranking) mang lại kết quả tốt hơn, đặc biệt ở chỉ số **Context Precision** và **Faithfulness**. Việc đưa các chunk tài liệu có mức độ tương thích ngữ nghĩa cao nhất lên đầu context giúp mô hình LLM trích xuất thông tin một cách chuẩn xác nhất, giảm thiểu hiện tượng ảo giác (hallucination) và nâng cao chất lượng câu trả lời.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | N/A | 0.77 | 0.83 | 0.40 | Retrieval | Chưa tìm đủ tài liệu làm căn cứ (Context Recall thấp) |
| 2 | N/A | 0.75 | 0.00 | 0.33 | Retrieval | Chưa tìm đủ tài liệu làm căn cứ (Context Recall thấp) |
| 3 | N/A | 0.85 | 0.83 | 0.60 | Generation | LLM generated answer without enough specific evidence |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng kích thước chunk size và overlap khi tách tài liệu (Task 4) đối với các tài liệu pháp luật (PDF/DOCX) để các điều khoản pháp luật không bị cắt đứt giữa chừng.  
**Expected impact:** Nâng cao điểm số Context Recall đối với các câu hỏi chi tiết về các điều khoản luật cụ thể.  

### Cải tiến 2
**Action:** Tinh chỉnh prompt hệ thống (SYSTEM_PROMPT) trong Task 10 để nhấn mạnh việc LLM chỉ được sử dụng dữ liệu từ context và nghiêm cấm tự suy đoán.  
**Expected impact:** Nâng điểm Faithfulness của cả 2 cấu hình lên tối đa.  

### Cải tiến 3
**Action:** Bổ sung cơ chế lọc nhiễu các thẻ HTML, liên kết thừa từ dữ liệu cào báo chí (Task 2) trước khi đưa vào lưu trữ.  
**Expected impact:** Nâng cao điểm số Context Precision của hệ thống.  
