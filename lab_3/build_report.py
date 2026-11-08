import json
from pathlib import Path

lines = []
for path in sorted(Path("results").glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"\n## {path.stem}\n")
    lines.append("| Назва | System | Prompt | Результат | Аналіз |")
    lines.append("|---|---|---|---|---|")
    for d in data:
        cell = lambda s: (s or "—").replace("\n", "<br>").replace("|", "\\|")
        lines.append(f"| {d['label']} | {cell(d['system'])} | {cell(d['prompt'])} | {cell(d['answer'])} | |")

Path("report.md").write_text("\n".join(lines), encoding="utf-8")
print("report.md створено")