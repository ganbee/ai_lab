"""Завдання 5. Пошук (retrieval) найрелевантніших фрагментів у ChromaDB.

Запит -> embedding (RETRIEVAL_QUERY) -> пошук top-k у векторному сховищі ->
виведення знайдених фрагментів, їхніх джерел і відстаней. LLM на цьому кроці не використовується.
Функція retrieve() повторно використовується в task4-task6.
"""
import json

import chromadb

from main import CHROMA_DIR, RESULTS_DIR, embed_texts

COLLECTION = "ergo_rag"
TOP_K = 3

# Запит і джерела, в яких ми очікуємо знайти відповідь (для оцінки якості пошуку)
QUERIES = [
    ("Що таке ерготерапія?", ["document_1.pdf", "document_3.docx"]),
    ("Які стратегії виділяють в ерготерапевтичному втручанні?", ["document_1.pdf"]),
    ("Що означають SMART-критерії при постановці цілей?", ["document_1.pdf"]),
    ("Які порушення виникають після інсульту і з чим працює ерготерапевт?", ["document_2.pdf"]),
    ("Як адаптують житло пацієнта після інсульту?", ["document_2.pdf"]),
    ("Який діапазон рухів у плечовому суглобі?", ["document_1.pdf"]),
    ("Яка ситуація з ерготерапією в Україні?", ["document_3.docx"]),
]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION)


def retrieve(query, k=TOP_K, collection=None):
    """Повертає k найближчих chunks до запиту: id, джерело, відстань, текст."""
    collection = collection or get_collection()
    query_vector = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    res = collection.query(query_embeddings=[query_vector], n_results=k)
    hits = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]),
            start=1):
        hits.append({
            "rank": rank,
            "chunk_id": cid,
            "source": meta["source"],
            "distance": round(dist, 4),
            "text": doc,
        })
    return hits


if __name__ == "__main__":
    collection = get_collection()
    print(f"У сховищі {collection.count()} записів\n")

    results, top1_ok, top3_ok = [], 0, 0
    for query, expected in QUERIES:
        hits = retrieve(query, collection=collection)
        in_top1 = hits[0]["source"] in expected
        in_top3 = any(h["source"] in expected for h in hits)
        top1_ok += in_top1
        top3_ok += in_top3

        print(f"Запит: {query}")
        print(f"Очікуване джерело: {', '.join(expected)}")
        for h in hits:
            print(f"  {h['rank']}. {h['chunk_id']} | {h['source']} | відстань {h['distance']}")
            print(f"     {h['text'][:130]}...")
        print()

        results.append({"query": query, "expected_sources": expected,
                        "top1_correct": in_top1, "top3_correct": in_top3, "hits": hits})

    n = len(QUERIES)
    print(f"Джерело з top-1 збіглося з очікуваним: {top1_ok}/{n}")
    print(f"Очікуване джерело є серед top-{TOP_K}: {top3_ok}/{n}")

    (RESULTS_DIR / "task3.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Збережено: results/task3.json")