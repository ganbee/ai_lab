"""Завдання 5. Порівняльний експеримент: зведена таблиця, N_correct, accuracy. API не викликається."""
import json

from main import RESULTS_DIR
from utils import load_test_set, normalize

APPROACHES = [("Prompt", "task2", "Prompt"), ("RAG", "task3", "RAG"), ("Fine-tuning (імітація)", "task4", "FewShot")]


def load_answers(log):
    path = RESULTS_DIR / f"{log}.json"
    if not path.exists():
        return {}
    return {d["label"]: d["answer"] for d in json.loads(path.read_text(encoding="utf-8"))}


def main():
    tests = load_test_set()
    answers = {name: load_answers(log) for name, log, _ in APPROACHES}
    rows = []
    totals = {name: {"all": 0, "clear": 0, "borderline": 0} for name, _, _ in APPROACHES}
    for t in tests:
        row = {"id": t["id"], "type": t["type"], "text": t["text"], "correct": t["correct"]}
        for name, _, prefix in APPROACHES:
            raw = answers[name].get(f"{prefix}-{t['id']}")
            pred = normalize(raw) if raw else "—"
            ok = pred == t["correct"]
            row[name] = {"answer": pred, "ok": ok}
            totals[name]["all"] += ok
            totals[name]["clear" if t["type"] == "чіткий" else "borderline"] += ok
        rows.append(row)

    n_all = len(tests)
    n_clear = sum(t["type"] == "чіткий" for t in tests)
    n_border = n_all - n_clear
    summary = {}
    for name, _, _ in APPROACHES:
        s = totals[name]
        summary[name] = {
            "n_correct": s["all"], "n_total": n_all,
            "accuracy": round(s["all"] / n_all * 100, 1),
            "accuracy_clear": round(s["clear"] / n_clear * 100, 1),
            "accuracy_borderline": round(s["borderline"] / n_border * 100, 1),
        }

    print(f"{'№':<3}{'Тип':<10}{'Еталон':<26}" + "".join(f"{n:<26}" for n, _, _ in APPROACHES))
    for r in rows:
        print(f"{r['id']:<3}{r['type']:<10}{r['correct']:<26}" +
              "".join(f"{(r[n]['answer'] + (' ✔' if r[n]['ok'] else ' ✘')):<26}" for n, _, _ in APPROACHES))
    print()
    for name, s in summary.items():
        print(f"{name}: {s['n_correct']}/{s['n_total']} = {s['accuracy']}% "
              f"(чіткі {s['accuracy_clear']}%, межові {s['accuracy_borderline']}%)")

    (RESULTS_DIR / "task5.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
