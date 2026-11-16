"""Завдання 6-7. Повний RAG: запит -> retrieval -> контекст -> Gemini -> відповідь.

Для кожного запитання знаходить top-k фрагментів, формує prompt за шаблоном з методички,
отримує відповідь Gemini (temperature = 0) і зберігає таблицю результатів у results/task4.json.
Тексти prompt і відповіді додатково логуються через ask() у results/task4_prompts.json.
Функція ask_rag() повторно використовується в task5-task6.
"""
import json

from main import RESULTS_DIR, ask
from task3 import QUERIES, TOP_K, get_collection, retrieve

LOG = "task4_prompts"

SYSTEM = (
    "Ти — асистент, який відповідає на запитання з ерготерапії та фізичної реабілітації. "
    "Відповідай українською мовою, стисло й лише на основі наданого контексту. "
    "Не додавай фактів, яких немає в контексті, і не використовуй власних знань. "
    "Якщо контекст не містить відповіді, прямо скажи, що в наданих документах цієї "
    "інформації немає. Наприкінці відповіді в дужках вкажи назви документів-джерел, "
    "з яких взято інформацію."
)

PROMPT_TEMPLATE = """Використовуючи наведений контекст, дай відповідь на запитання користувача.

Контекст:
{context}

Запитання:
{question}

Якщо відповіді на запитання немає в контексті, повідом про це."""


def build_context(hits):
    """Склеює знайдені фрагменти в один контекст із позначкою джерела."""
    return "\n\n".join(
        f"[{h['rank']}] (джерело: {h['source']}, {h['chunk_id']})\n{h['text']}"
        for h in hits
    )


def generate_answer(question, hits, log=LOG, label=None):
    """Передає знайдені фрагменти Gemini як контекст і повертає відповідь."""
    prompt = PROMPT_TEMPLATE.format(context=build_context(hits), question=question)
    return ask(log, label or question, prompt, system=SYSTEM, temperature=0.0)


def ask_rag(question, k=TOP_K, collection=None, log=LOG, label=None):
    """Повний ланцюжок RAG: повертає (знайдені фрагменти, відповідь)."""
    hits = retrieve(question, k=k, collection=collection)
    answer = generate_answer(question, hits, log=log, label=label)
    return hits, answer


if __name__ == "__main__":
    collection = get_collection()
    rows = []
    for n, (question, _) in enumerate(QUERIES, start=1):
        hits, answer = ask_rag(question, collection=collection, label=f"Q{n}: {question}")
        rows.append({
            "n": n,
            "query": question,
            "chunks": [{"chunk_id": h["chunk_id"], "source": h["source"],
                        "distance": h["distance"], "preview": h["text"][:200]}
                       for h in hits],
            "sources": sorted({h["source"] for h in hits}),
            "answer": answer,
        })

    (RESULTS_DIR / "task4.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for r in rows:
        print(f"{r['n']}. {r['query']}")
        print(f"   Джерела: {', '.join(r['sources'])}")
        print(f"   Chunks: {', '.join(c['chunk_id'] for c in r['chunks'])}")
    print("\nЗбережено: results/task4.json, results/task4_prompts.json")