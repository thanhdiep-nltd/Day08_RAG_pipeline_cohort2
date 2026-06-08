"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import weaviate
from weaviate.classes.query import MetadataQuery

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model ở Task 4 (text-embedding-3-small)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing from environment variables.")

    openai_client = OpenAI(api_key=openai_key)
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # Bước 2: Query vector store (cosine similarity)
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    if not weaviate_url or not weaviate_key:
        raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY is missing from environment variables.")

    results_list = []
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_key),
        skip_init_checks=True
    ) as client:
        collection_name = "DrugLawDocs"
        if not client.collections.exists(collection_name):
            print(f"  [WARNING] Collection '{collection_name}' does not exist on Weaviate.")
            return []

        collection = client.collections.get(collection_name)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )

        # Bước 3: Return top_k results có tính score từ distance
        for obj in results.objects:
            # Weaviate distance defaults to cosine distance for cosine metric
            # Cosine similarity = 1 - cosine distance
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 0.0
            score = 1.0 - distance

            results_list.append({
                "content": obj.properties.get("content", ""),
                "score": float(score),
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "doc_type": obj.properties.get("doc_type", ""),
                    "chunk_index": int(obj.properties.get("chunk_index", 0))
                }
            })

    # Đảm bảo kết quả được sắp xếp giảm dần theo điểm tương đồng
    results_list.sort(key=lambda x: x["score"], reverse=True)
    return results_list


if __name__ == "__main__":
    # Test
    print("=" * 50)
    print("Testing Semantic Search...")
    print("=" * 50)
    results = semantic_search("dùng nhánh tỏi để đuổi ma", top_k=5)
    for i, r in enumerate(results, 1):
        source = r['metadata']['source']
        chunk_index = r['metadata']['chunk_index']
        print(f"{i}. [{r['score']:.4f}] ({source} - Index: {chunk_index})")
        # Chuyển sang ký tự an toàn để in ra màn hình console Windows
        safe_content = r['content'][:150].encode("ascii", "replace").decode("ascii")
        print(f"   Content: {safe_content}...")
        print("-" * 50)
