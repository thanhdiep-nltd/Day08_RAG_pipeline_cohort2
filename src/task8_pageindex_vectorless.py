"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


import time
from pageindex import PageIndexClient
from fpdf import FPDF

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.pdf"
DOC_ID_CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_id.txt"


def upload_documents():
    """
    Tải toàn bộ tài liệu (dưới dạng PDF kết hợp) lên PageIndex.
    """
    pageindex_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not pageindex_key or pageindex_key.startswith("pi_"):
        raise ValueError("PAGEINDEX_API_KEY chưa được cấu hình hợp lệ trong file .env!")

    # Bước 1: Tạo file PDF kết hợp từ các file markdown
    print("  Generating combined PDF from standardized markdown documents...")
    pdf = FPDF()
    pdf.add_page()
    
    # Sử dụng font hệ thống có sẵn trên Windows để hiển thị tiếng Việt Unicode
    pdf.add_font("Arial", "", "C:/Windows/Fonts/Arial.ttf")
    pdf.add_font("ArialBold", "", "C:/Windows/Fonts/Arialbd.ttf")

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        
        pdf.set_font("ArialBold", size=12)
        pdf.cell(0, 10, text=f"--- DOCUMENT: {md_file.name} ({doc_type}) ---", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Arial", size=10)
        
        clean_content = content.replace("\r", "")
        pdf.multi_cell(0, 5, text=clean_content)
        pdf.cell(0, 10, text="", new_x="LMARGIN", new_y="NEXT")
        
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_PATH))
    print(f"  ✓ Saved combined PDF at: {PDF_PATH}")

    # Bước 2: Khởi tạo client và tải tài liệu lên PageIndex
    client = PageIndexClient(api_key=pageindex_key)
    print("  Uploading PDF to PageIndex...")
    upload_res = client.submit_document(str(PDF_PATH))
    doc_id = upload_res.get("doc_id")
    if not doc_id:
        raise RuntimeError(f"Tải tài liệu thất bại, API phản hồi: {upload_res}")
        
    print(f"  ✓ Uploaded document, received doc_id: {doc_id}")
    
    # Bước 3: Chờ tài liệu được xử lý xong
    print("  Waiting for document indexing status 'completed'...")
    while True:
        try:
            if client.is_retrieval_ready(doc_id):
                print("    - Status: ready for retrieval!")
                break
            doc_info = client.get_document(doc_id)
            if "status" not in doc_info:
                raise RuntimeError(f"PageIndex document check failed: {doc_info}")
            status = doc_info.get("status")
            print(f"    - Status: {status}")
            if status == "failed":
                raise RuntimeError("PageIndex xử lý tài liệu thất bại!")
            elif status not in ("queued", "processing"):
                raise RuntimeError(f"PageIndex document check returned unexpected status: {status}")
        except Exception as e:
            print(f"    - Error checking status: {e}")
            raise e
        time.sleep(5)

    # Bước 4: Lưu doc_id vào cache
    DOC_ID_CACHE_FILE.write_text(doc_id, encoding="utf-8")
    print(f"  ✓ Indexed successfully! Cache saved at: {DOC_ID_CACHE_FILE}")
    return doc_id


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    pageindex_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not pageindex_key or pageindex_key.startswith("pi_"):
        raise ValueError("PAGEINDEX_API_KEY chưa cấu hình hoặc không hợp lệ.")

    # Đọc doc_id từ cache
    if not DOC_ID_CACHE_FILE.exists():
        print("  [INFO] Không tìm thấy cache doc_id. Tiến hành biên dịch và tải tài liệu mới...")
        doc_id = upload_documents()
    else:
        doc_id = DOC_ID_CACHE_FILE.read_text(encoding="utf-8").strip()

    if not doc_id:
        return []

    client = PageIndexClient(api_key=pageindex_key)
    
    print(f"  Submitting query to PageIndex (doc_id: {doc_id})...")
    query_res = client.submit_query(doc_id, query)
    retrieval_id = query_res.get("retrieval_id")
    if not retrieval_id:
        raise RuntimeError(f"Gửi query thất bại, API phản hồi: {query_res}")

    # Polling kết quả query
    print("  Waiting for query retrieval to complete...")
    while True:
        ret_res = client.get_retrieval(retrieval_id)
        if "status" not in ret_res:
            raise RuntimeError(f"PageIndex retrieval failed: {ret_res}")
        status = ret_res.get("status")
        if status == "completed":
            break
        elif status == "failed":
            raise RuntimeError("PageIndex retrieval failed!")
        elif status not in ("processing", "queued", "running"):
            raise RuntimeError(f"PageIndex retrieval returned unexpected status: {status}")
        time.sleep(1)

    # Trích xuất các nodes kết quả
    retrieved_nodes = ret_res.get("retrieved_nodes", [])
    results = []
    
    for node in retrieved_nodes:
        node_title = node.get("title", "")
        relevant_contents = node.get("relevant_contents", [])
        for rc in relevant_contents:
            text = rc.get("relevant_content", "")
            page_index = rc.get("page_index", 0)
            
            # PageIndex không trả về điểm cosine, đặt điểm tương quan mặc định là 1.0
            results.append({
                "content": text,
                "score": 1.0,
                "metadata": {
                    "source": "combined_rag_documents.pdf",
                    "node_title": node_title,
                    "page_index": page_index
                },
                "source": "pageindex"
            })
            
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_"):
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        try:
            upload_documents()
            print("\nTest query:")
            results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
            for r in results:
                print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        except Exception as e:
            print(f"[ERROR] PageIndex process failed: {e}")
