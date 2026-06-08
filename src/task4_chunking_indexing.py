"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import os
import json
from dotenv import load_dotenv

# Load .env từ thư mục gốc dự án
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn chunking strategy: "recursive" để đảm bảo các chunk phân bổ đồng đều, giữ được câu nguyên vẹn.
# Kích thước 800 ký tự (~200-250 từ) là độ dài tối ưu cho model text-embedding-3-small của OpenAI.
# Overlap 100 ký tự để bảo toàn mối liên hệ ngữ cảnh giữa các chunk.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn embedding model: OpenAI text-embedding-3-small (1536 dim) vì chất lượng tốt, chi phí API rẻ
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Chọn vector store: Weaviate Cloud (Free Sandbox) hỗ trợ hybrid search tích hợp sẵn
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"  [ERROR] Standardized directory not found: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    texts = [c["content"] for c in chunks]

    print(f"  Generating embeddings for {len(texts)} chunks using {EMBEDDING_MODEL}...")
    
    # Chia batch 100 để tránh lỗi giới hạn hoặc tối ưu tốc độ gọi API
    batch_size = 100
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )
        embeddings.extend([item.embedding for item in response.data])

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import weaviate
    from weaviate.classes.config import Property, DataType, Configure

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_key:
        raise ValueError("Thiếu WEAVIATE_URL hoặc WEAVIATE_API_KEY trong file .env!")

    print(f"  Connecting to Weaviate Cloud at {weaviate_url}...")
    
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_key)
    ) as client:
        collection_name = "DrugLawDocs"

        # Khởi tạo/Làm sạch collection
        if client.collections.exists(collection_name):
            print(f"  Cleaning existing collection: {collection_name}...")
            client.collections.delete(collection_name)

        print(f"  Creating new collection: {collection_name}...")
        collection = client.collections.create(
            name=collection_name,
            vector_config=Configure.Vectorizer.none(),  # Tự truyền vector đã tính từ OpenAI
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )

        # Ghi hàng loạt (batch insert) lên Cloud
        print(f"  Uploading {len(chunks)} chunks to Weaviate Cloud...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )

        if collection.batch.failed_objects:
            print(f"  [ERROR] Failed to insert {len(collection.batch.failed_objects)} objects")
            for failed in collection.batch.failed_objects:
                print(f"    - Error message: {failed.message}")
        else:
            print(f"  [OK] Successfully indexed all {len(chunks)} chunks to collection '{collection_name}' on Weaviate Cloud.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Finished Task 4 pipeline execution.")


if __name__ == "__main__":
    run_pipeline()
