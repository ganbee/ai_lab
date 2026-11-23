"""Завдання 3. RAG: пошук релевантних правил у базі знань і класифікація за знайденим контекстом."""
import hashlib
import json

import chromadb

from main import ask, embed_texts, DATA_DIR, RESULTS_DIR, CHROMA_DIR
from utils import CATEGORIES, load_test_set, normalize

KB_DIR = DATA_DIR / "kb"
COLLECTION = "lab6_kb"  # окрема колекція, щоб не чіпати базу з лаб. 5
TOP_K = 2
SOURCE_TO_CATEGORY = {
    "adl.md": "повсякденна діяльність",
    "hand_function.md": "функція руки",
    "cognitive.md": "когнітивна реабілітація",
    "adaptation.md": "адаптація середовища",
}
QUERY_CACHE = RESULTS_DIR / "task3_query_emb.json"

PROMPT_TEMPLATE = """Ти — фахівець з ерготерапії.
Нижче наведено фрагменти з бази знань про категорії втручань.

Контекст:
{context}

Користуючись контекстом, визнач, до якого напряму ерготерапевтичного втручання належить ситуація.
Можливі категорії: {categories}.
Поверни лише назву категорії.

Ситуація: {text}"""


def load_chunks():
    """Кожен абзац документа — окремий chunk із назвою документа на початку."""
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        title = paragraphs[0].lstrip("# ").strip()
        for i, p in enumerate(paragraphs[1:], start=1):
            chunks.append({"id": f"{path.stem}_{i}", "source": path.name, "text": f"{title}. {p}"})
    return chunks


def build_index(chunks):
    """Будує колекцію ChromaDB; якщо база знань не змінилась, повторно не вбудовує."""
    kb_hash = hashlib.md5("".join(c["text"] for c in chunks).encode("utf-8")).hexdigest()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        col = client.get_collection(COLLECTION)
        if col.metadata and col.metadata.get("kb_hash") == kb_hash:
            print(f"Індекс актуальний ({col.count()} chunks), embeddings не перераховуємо")
            return col
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "kb_hash": kb_hash})
    vectors = embed_texts([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Збережено {col.count()} chunks")
    return col


def embed_queries(texts):
    """Embeddings запитів із кешем (економія квоти при повторних запусках)."""
    cache = json.loads(QUERY_CACHE.read_text(encoding="utf-8")) if QUERY_CACHE.exists() else {}
    missing = [t for t in texts if t not in cache]
    if missing:
        for t, v in zip(missing, embed_texts(missing, "RETRIEVAL_QUERY")):
            cache[t] = v
        QUERY_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[t] for t in texts]


def main():
    chunks = load_chunks()
    print(f"Chunks: {len(chunks)}, макс. довжина: {max(len(c['text']) for c in chunks)} символів")
    col = build_index(chunks)

    tests = load_test_set()
    q_vectors = embed_queries([t["text"] for t in tests])

    retrieval_log, correct = [], 0
    for t, vec in zip(tests, q_vectors):
        res = col.query(query_embeddings=[vec], n_results=TOP_K,
                        include=["documents", "metadatas", "distances"])
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        sources = [m["source"] for m in metas]
        expected_source = next(s for s, c in SOURCE_TO_CATEGORY.items() if c == t["correct"])
        retrieval_log.append({
            "id": t["id"],
            "expected_source": expected_source,
            "retrieved": [{"source": s, "distance": round(d, 4)} for s, d in zip(sources, dists)],
            "top1_correct": sources[0] == expected_source,
            "in_top_k": expected_source in sources,
        })

        context = "\n".join(f"[{i}] {d}" for i, d in enumerate(docs, start=1))
        prompt = PROMPT_TEMPLATE.format(context=context, categories=", ".join(CATEGORIES), text=t["text"])
        answer = ask("task3", f"RAG-{t['id']}", prompt, reuse=True)
        ok = normalize(answer) == t["correct"]
        correct += ok
        print(f"#{t['id']}: еталон={t['correct']} | модель={normalize(answer)} | "
              f"пошук top1={'✔' if retrieval_log[-1]['top1_correct'] else '✘'} | {'✔' if ok else '✘'}")

    (RESULTS_DIR / "task3_retrieval.json").write_text(
        json.dumps(retrieval_log, ensure_ascii=False, indent=2), encoding="utf-8")
    top1 = sum(r["top1_correct"] for r in retrieval_log)
    print(f"\nПошук: top-1 правильний у {top1}/{len(tests)}")
    print(f"RAG: {correct}/{len(tests)} = {correct / len(tests) * 100:.1f}%")


if __name__ == "__main__":
    main()