"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (Union/Weighted Score) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Bước 1: Chạy semantic_search và lexical_search
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  [WARNING] Semantic search failed: {e}")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  [WARNING] Lexical search failed: {e}")
        sparse_results = []

    # Bước 2: Gộp kết quả loại bỏ trùng lặp và lấy score lớn nhất
    merged_map = {}
    for item in dense_results + sparse_results:
        content = item["content"]
        if content not in merged_map:
            merged_map[content] = item.copy()
        else:
            if item["score"] > merged_map[content]["score"]:
                merged_map[content]["score"] = item["score"]

    merged = list(merged_map.values())
    for item in merged:
        item["source"] = "hybrid"

    # Bước 3: Áp dụng rerank
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except Exception as e:
            print(f"  [WARNING] Reranking failed: {e}. Fallback to sorting by retrieval scores.")
            merged.sort(key=lambda x: x["score"], reverse=True)
            final_results = merged[:top_k]
    else:
        merged.sort(key=lambda x: x["score"], reverse=True)
        final_results = merged[:top_k]

    # Đảm bảo trường source được gán đúng
    for item in final_results:
        item["source"] = "hybrid"

    # Bước 4: Kiểm tra ngưỡng score để chuyển hướng fallback PageIndex
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(f"  [INFO] Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). Fallback -> PageIndex")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as e:
            print(f"  [WARNING] PageIndex search failed or skipped: {e}")

    return final_results


if __name__ == "__main__":
    import sys
    # Reconfigure stdout to use utf-8 to avoid UnicodeEncodeError on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
