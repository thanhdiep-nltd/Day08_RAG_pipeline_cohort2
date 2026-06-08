"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import string
import re
from pathlib import Path

# Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_BM25_INDEX = None


def load_corpus_if_empty():
    """Tải dữ liệu từ thư mục standardized và phân mảnh (chunking) nếu CORPUS chưa được nạp."""
    global CORPUS
    if not CORPUS:
        try:
            # Ưu tiên import dạng package đầy đủ từ root
            from src.task4_chunking_indexing import load_documents, chunk_documents
            docs = load_documents()
            CORPUS.extend(chunk_documents(docs))
        except ImportError:
            # Fallback nếu chạy trực tiếp script hoặc chạy từ thư mục src/
            import sys
            src_dir = Path(__file__).parent
            if str(src_dir) not in sys.path:
                sys.path.append(str(src_dir))
            from task4_chunking_indexing import load_documents, chunk_documents
            docs = load_documents()
            CORPUS.extend(chunk_documents(docs))


def tokenize(text: str) -> list[str]:
    """Tokenize văn bản (chuyển chữ thường, bỏ dấu câu và phân tách bằng khoảng trắng)."""
    text = text.lower()
    # Loại bỏ các ký tự dấu câu
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text.split()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def get_bm25_index():
    global _BM25_INDEX
    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(CORPUS)
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    load_corpus_if_empty()
    if not CORPUS:
        return []

    bm25 = get_bm25_index()
    tokenized_query = tokenize(query)
    
    # Tính điểm BM25 cho các tài liệu
    scores = bm25.get_scores(tokenized_query)
    
    # Lấy các index được sắp xếp giảm dần theo điểm số
    import numpy as np
    top_indices = np.argsort(scores)[::-1]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
            
    return results[:top_k]


if __name__ == "__main__":
    # Test
    print("=" * 50)
    print("Testing Lexical Search (BM25)...")
    print("=" * 50)
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for i, r in enumerate(results, 1):
        # Tránh UnicodeEncodeError khi chạy trên CMD của Windows
        safe_content = r['content'][:120].encode("ascii", "replace").decode("ascii")
        print(f"{i}. [{r['score']:.3f}] (Source: {r['metadata']['source']})")
        print(f"   Content: {safe_content}...")
        print("-" * 50)
