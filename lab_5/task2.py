"""Завдання 3-4. Embeddings для chunks і запис у векторне сховище ChromaDB.

Читає results/task1.json, створює embedding для кожного chunk (RETRIEVAL_DOCUMENT),
зберігає в ChromaDB (текст + вектор + джерело) та пише підсумок у results/task2.json.
Усі вектори додатково кешуються у results/embeddings.json, щоб не витрачати квоту API
при повторному запуску.
"""
import json

import chromadb

from main import CHROMA_DIR, EMBED_DIM, EMBED_MODEL, RESULTS_DIR, embed_texts

COLLECTION = "ergo_rag"
CACHE = RESULTS_DIR / "embeddings.json"


def load_chunks():
    return json.loads((RESULTS_DIR / "task1.json").read_text(encoding="utf-8"))


def get_vectors(chunks):
    """Бере вектори з кешу, якщо він відповідає поточним chunks; інакше рахує через API."""
    if CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if (cached["model"] == EMBED_MODEL and cached["dim"] == EMBED_DIM
                and cached["ids"] == [c["chunk_id"] for c in chunks]):
            print("Беру embeddings із кешу results/embeddings.json")
            return cached["vectors"]

    vectors = embed_texts([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    CACHE.write_text(json.dumps({
        "model": EMBED_MODEL,
        "dim": EMBED_DIM,
        "ids": [c["chunk_id"] for c in chunks],
        "vectors": vectors,
    }), encoding="utf-8")
    return vectors


def build_vector_store(chunks, vectors):
    """Створює колекцію ChromaDB заново і заносить у неї chunks."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_in_doc": c["chunk_in_doc"]}
                   for c in chunks],
    )
    return collection


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Chunks: {len(chunks)}")

    vectors = get_vectors(chunks)
    print(f"Розмірність embedding: {len(vectors[0])}")

    collection = build_vector_store(chunks, vectors)
    print(f"У ChromaDB записано: {collection.count()} записів")

    example = chunks[0]
    vec = vectors[0]
    print(f"\n{example['chunk_id']} ({example['source']}):")
    print(example["text"][:150], "...")
    print("Embedding:", [round(x, 3) for x in vec[:6]], "...")

    summary = {
        "embedding_model": EMBED_MODEL,
        "dimension": len(vectors[0]),
        "task_type_documents": "RETRIEVAL_DOCUMENT",
        "vector_store": "ChromaDB",
        "collection": COLLECTION,
        "distance": "cosine",
        "n_records": collection.count(),
        "example": {
            "chunk_id": example["chunk_id"],
            "source": example["source"],
            "text": example["text"],
            "embedding_first_10": vec[:10],
            "embedding_length": len(vec),
        },
    }
    (RESULTS_DIR / "task2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nЗбережено: results/task2.json")