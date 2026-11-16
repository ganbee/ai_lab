"""Додаткове завдання. Порівняння відповідей LLM без RAG і з RAG.

Ті самі запитання ставляться Gemini двічі: напряму (без контексту) і через RAG (task4.ask_rag).
Обрано запитання, відповіді на які в документах є конкретні (числа, переліки), тож
правильність видно одразу. Для кожної відповіді рахується груба автоперевірка наявності
ключових фактів; остаточну оцінку точності, повноти й вигадок виставляємо вручну.
Результати: results/task6.json (+ results/task6_norag.json, task6_rag.json через ask()).
"""
import json

from main import RESULTS_DIR, ask
from task3 import get_collection
from task4 import ask_rag

LOG_NORAG = "task6_norag"
LOG_RAG = "task6_rag"

# Системна інструкція без RAG: та сама роль і мова, але без вимоги спиратися на контекст
SYSTEM_NORAG = (
    "Ти — асистент, який відповідає на запитання з ерготерапії та фізичної реабілітації. "
    "Відповідай українською мовою стисло."
)

# (запитання, еталонна відповідь за документами, ключові фрагменти для автоперевірки)
QUESTIONS = [
    ("Які стратегії виділяють в ерготерапевтичному втручанні?",
     "Розвиваюча, з відновленням функції, компенсаторна (document_1.pdf)",
     ["розвиваюч", "компенсаторн", "відновленням функції"]),
    ("Яка рекомендована тривалість і частота занять із тренування навичок "
     "повсякденної активності після інсульту?",
     "20–40 хв, щодня або через день (document_2.pdf)",
     ["20-40"]),
    ("У якої частки пацієнтів із наслідками травм хребта і спинного мозку "
     "відновлюється працездатність?",
     "Повністю або частково лише в 2–10 % пацієнтів (document_1.pdf)",
     ["2-10"]),
    ("Скільки людей у світі щороку переносять інсульт і скільки з них помирає?",
     "16 млн щороку, з них 6 млн помирає (document_2.pdf)",
     ["16 млн", "6 млн"]),
]


def normalize(text):
    """Уніфікує тире й пробіли, щоб '20–40' і '20 - 40' рахувались однаково."""
    t = text.lower().replace("–", "-").replace("—", "-").replace("−", "-")
    t = t.replace(" - ", "-").replace("\xa0", " ")
    return t


def has_keys(answer, keys):
    t = normalize(answer)
    return all(normalize(k) in t for k in keys)


if __name__ == "__main__":
    collection = get_collection()
    rows = []
    for n, (question, reference, keys) in enumerate(QUESTIONS, start=1):
        no_rag = ask(LOG_NORAG, f"Q{n} без RAG: {question}", question,
                     system=SYSTEM_NORAG, temperature=0.0, reuse=True)
        hits, rag = ask_rag(question, collection=collection, log=LOG_RAG,
                            label=f"Q{n} з RAG: {question}")
        rows.append({
            "n": n,
            "query": question,
            "reference": reference,
            "answer_without_rag": no_rag,
            "answer_with_rag": rag,
            "rag_chunks": [{"chunk_id": h["chunk_id"], "source": h["source"],
                            "distance": h["distance"]} for h in hits],
            "key_facts_in_answer_without_rag": has_keys(no_rag, keys),
            "key_facts_in_answer_with_rag": has_keys(rag, keys),
        })

    (RESULTS_DIR / "task6.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for r in rows:
        print(f"{r['n']}. {r['query']}")
        print(f"   Еталон: {r['reference']}")
        print(f"   Ключові факти: без RAG = {r['key_facts_in_answer_without_rag']}, "
              f"з RAG = {r['key_facts_in_answer_with_rag']}")
    n = len(rows)
    print(f"\nБез RAG: {sum(r['key_facts_in_answer_without_rag'] for r in rows)}/{n}, "
          f"з RAG: {sum(r['key_facts_in_answer_with_rag'] for r in rows)}/{n}")
    print("Збережено: results/task6.json")