"""Збирає таблиці для звіту (пункти 7-9) у results/report_tables.md."""
import json

from main import RESULTS_DIR

CRITERIA = {
    "Простота реалізації":                    {"Prompt": 5, "RAG": 2, "Fine-tuning (імітація)": 4},
    "Необхідні ресурси":                      {"Prompt": 5, "RAG": 2, "Fine-tuning (імітація)": 4},
    "Швидкість підготовки":                   {"Prompt": 5, "RAG": 2, "Fine-tuning (імітація)": 3},
    "Можливість роботи з новими даними":      {"Prompt": 2, "RAG": 5, "Fine-tuning (імітація)": 2},
    "Контроль результату":                    {"Prompt": 3, "RAG": 4, "Fine-tuning (імітація)": 4},
}


def quality_score(accuracy):
    return max(1, min(5, round(accuracy / 20)))


def main():
    data = json.loads((RESULTS_DIR / "task5.json").read_text(encoding="utf-8"))
    rows, summary = data["rows"], data["summary"]
    names = list(summary)
    out = []

    out.append("## Таблиця 1. Результати на тестових даних\n")
    out.append("| № | Тип | Вхідні дані | Правильна відповідь | " + " | ".join(names) + " |")
    out.append("|---|---|---|---|" + "---|" * len(names))
    for r in rows:
        cells = [f"{r[n]['answer']} {'✔' if r[n]['ok'] else '✘'}" for n in names]
        out.append(f"| {r['id']} | {r['type']} | {r['text']} | {r['correct']} | " + " | ".join(cells) + " |")

    out.append("\n## Таблиця 2. Точність\n")
    out.append("| Підхід | N_correct / N_total | Accuracy | Чіткі | Межові |")
    out.append("|---|---|---|---|---|")
    for n in names:
        s = summary[n]
        out.append(f"| {n} | {s['n_correct']}/{s['n_total']} | {s['accuracy']}% | "
                   f"{s['accuracy_clear']}% | {s['accuracy_borderline']}% |")

    retr_path = RESULTS_DIR / "task3_retrieval.json"
    if retr_path.exists():
        retr = json.loads(retr_path.read_text(encoding="utf-8"))
        top1 = sum(r["top1_correct"] for r in retr)
        topk = sum(r["in_top_k"] for r in retr)
        out.append(f"\nТочність пошуку RAG: правильний документ першим — {top1}/{len(retr)}, "
                   f"серед top-k — {topk}/{len(retr)}.")

    out.append("\n## Таблиця 3. Оцінювання підходів (1-5)\n")
    out.append("| Критерій | " + " | ".join(names) + " |")
    out.append("|---|" + "---|" * len(names))
    out.append("| Якість результату | " + " | ".join(str(quality_score(summary[n]["accuracy"])) for n in names) + " |")
    for crit, scores in CRITERIA.items():
        out.append(f"| {crit} | " + " | ".join(str(scores[n]) for n in names) + " |")

    path = RESULTS_DIR / "report_tables.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Збережено {path}")


if __name__ == "__main__":
    main()