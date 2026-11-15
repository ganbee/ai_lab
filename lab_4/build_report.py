import json
from pathlib import Path

RESULTS = Path("results")

SECTIONS = [
    ("task1", "Завдання 1. Простий і просунутий prompt"),
    ("task2", "Завдання 2. Декомпозиція складної задачі"),
    ("task3", "Завдання 3. Імітація ReAct"),
    ("task4", "Завдання 4. Структурований результат (JSON)"),
    ("task5", "Завдання 5. Великий контекст і chunking"),
    ("task6", "Завдання 6. Параметри генерації (temperature, top-p)"),
]

out = []
out.append("# Лабораторна робота № 4\n")
out.append("**Тема:** Просунутий Prompt Engineering, Context Engineering та керування генерацією\n")
out.append("**Предметна область:** фізична терапія та реабілітація\n")
out.append("**Мета роботи:** *(вставити з методички)*\n")

model = "—"
for name, _ in SECTIONS:
    p = RESULTS / f"{name}.json"
    if p.exists():
        items = json.loads(p.read_text(encoding="utf-8"))
        if items:
            model = items[0].get("model", "—")
            break
out.append(f"## Використана LLM\n\nGemini, модель `{model}` (Google Gemini API, бібліотека google-genai).\n")

for name, title in SECTIONS:
    out.append(f"## {title}\n")
    p = RESULTS / f"{name}.json"
    if not p.exists():
        out.append("*Результатів ще немає.*\n")
        continue
    for it in json.loads(p.read_text(encoding="utf-8")):
        out.append(f"### {it['label']}\n")
        out.append(f"Параметри: temperature = {it.get('temperature')}, top_p = {it.get('top_p')}\n")
        if it.get("system"):
            out.append(f"**System:**\n```\n{it['system']}\n```\n")
        out.append(f"**Prompt:**\n```\n{it['prompt']}\n```\n")
        out.append(f"**Відповідь:**\n\n{it['answer']}\n")

    extra = RESULTS / f"{name}_check.json"
    if extra.exists():
        out.append("**Перевірка структури (Python):**\n```json\n"
                   + extra.read_text(encoding="utf-8") + "\n```\n")
    for suffix in ("stats", "metrics"):
        extra = RESULTS / f"{name}_{suffix}.json"
        if extra.exists():
            out.append(f"**{suffix}:**\n```json\n" + extra.read_text(encoding="utf-8") + "\n```\n")

out.append("## Підсумкова таблиця (оцінки 1-5)\n")
out.append("| Підхід | Якість | Повнота | Стабільність | Креативність |\n"
           "|---|---|---|---|---|\n"
           "| Простий prompt | | | | |\n"
           "| Просунутий prompt | | | | |\n"
           "| Декомпозиція | | | | |\n"
           "| Chunking | | | | |\n"
           "| Temperature = 0 | | | | |\n"
           "| Temperature = 0.5 | | | | |\n"
           "| Temperature = 1 | | | | |\n")
out.append("## Висновки\n\n*(заповнити)*\n")
out.append("## Відповіді на контрольні запитання\n\n*(заповнити)*\n")

Path("report.md").write_text("\n".join(out), encoding="utf-8")
print("Готово: report.md")