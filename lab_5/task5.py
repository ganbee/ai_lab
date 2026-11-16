"""Завдання 8. Дослідження обмежень RAG: запитання, відповіді на які немає в базі.

Для кожного запитання, якого немає в документах, виконується повний RAG (task4.ask_rag).
Перевіряємо: чи визнає модель відсутність інформації, чи вигадує відповідь,
і чи можна було б відсіяти такий запит за відстанню до найближчого chunk (поріг схожості).
Результати: results/task5.json (і results/task5_prompts.json через ask()).
"""
import json

from main import RESULTS_DIR
from task3 import get_collection
from task4 import ask_rag

LOG = "task5_prompts"

# (тип запитання, текст). У корпусі немає ні метформіну, ні цін, ні препаратів при інсульті;
# слово «діабет» трапляється лише в одному chunk (chunk_053), але в іншому контексті.
OUT_OF_BASE = [
    ("Повністю поза темою", "Яка столиця Австралії?"),
    ("Суміжна медична тема", "Яка добова доза метформіну при цукровому діабеті 2 типу?"),
    ("Тема з бази, але деталі відсутні", "Скільки коштує сеанс ерготерапії в Україні?"),
    ("Тема з бази, але деталі відсутні",
     "Які препарати призначають у гострому періоді інсульту для захисту нейронів?"),
]

# Маркери того, що модель визнала відсутність інформації (груба евристика; остаточно оцінюємо вручну)
ABSENCE_MARKERS = ("немає", "не містить", "відсутн", "не згадується", "не наведено")


def in_base_top1_range():
    """Діапазон відстаней до найближчого chunk для запитань, що мають відповідь у базі (task3)."""
    data = json.loads((RESULTS_DIR / "task3.json").read_text(encoding="utf-8"))
    d = [item["hits"][0]["distance"] for item in data]
    return min(d), max(d)


if __name__ == "__main__":
    collection = get_collection()
    low, high = in_base_top1_range()
    print(f"Відстань top-1 для запитань ІЗ бази (task3): від {low} до {high}\n")

    rows = []
    for n, (kind, question) in enumerate(OUT_OF_BASE, start=1):
        hits, answer = ask_rag(question, collection=collection, log=LOG,
                               label=f"OOB{n}: {question}")
        best = hits[0]["distance"]
        rows.append({
            "n": n,
            "type": kind,
            "query": question,
            "best_distance": best,
            "above_in_base_max": best > high,
            "chunks": [{"chunk_id": h["chunk_id"], "source": h["source"],
                        "distance": h["distance"], "preview": h["text"][:200]}
                       for h in hits],
            "answer": answer,
            "says_no_info_heuristic": any(m in answer.lower() for m in ABSENCE_MARKERS),
        })

    out = {"in_base_top1_distance_range": [low, high], "questions": rows}
    (RESULTS_DIR / "task5.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for r in rows:
        print(f"{r['n']}. [{r['type']}] {r['query']}")
        print(f"   Найближчий chunk: {r['chunks'][0]['chunk_id']}, відстань {r['best_distance']}"
              f" ({'ВИЩЕ' if r['above_in_base_max'] else 'не вище'} за максимум для запитань із бази)")
        print(f"   Модель визнала відсутність інформації (евристика): {r['says_no_info_heuristic']}")
    print("\nЗбережено: results/task5.json, results/task5_prompts.json")