"""Тестування агента та збірка таблиці для звіту.

python run_tests.py            — виконати тести, яких ще немає в results/tests.json, і зібрати таблицю
python run_tests.py --force    — виконати всі тести заново
python run_tests.py --only 3   — виконати лише тест №3
python run_tests.py --report   — лише зібрати таблицю з наявних результатів

Результати: results/tests.json (сирі дані), results/tests_table.md (таблиця й повні діалоги).
"""
import argparse
import json
import time

from agent import MODEL, RESULTS_DIR, Memory, run_agent

OUT = RESULTS_DIR / "tests.json"
TABLE = RESULTS_DIR / "tests_table.md"

TESTS = [
    {"id": 1, "type": "Просте питання без інструмента",
     "queries": ["Що таке ерготерапія? Відповідь дай у трьох реченнях."],
     "expected_tools": []},
    {"id": 2, "type": "Задача з інструментом",
     "queries": ["Амплітуда згинання в лікті пацієнта: 80, 85, 92, 97 градусів. "
                 "Знайди середнє значення та відсоток покращення від першого виміру до останнього."],
     "expected_tools": ["calculator"]},
    {"id": 3, "type": "Багатокрокове завдання",
     "queries": ["Підбери вправи для дрібної моторики кисті початкового рівня і склади графік "
                 "занять на 3 тижні, по 3 заняття на тиждень, починаючи з 1 грудня 2026."],
     "expected_tools": ["search_exercises", "make_schedule"]},
    {"id": 4, "type": "Використання памʼяті (3 повідомлення)",
     "queries": ["Я працюю з пацієнтом після інсульту, у нього слабка кисть.",
                 "Які вправи ти порадиш?",
                 "Склади графік на 2 тижні, по 2 заняття на тиждень, починаючи з 7 грудня 2026."],
     "expected_tools": ["search_exercises", "make_schedule"],
     "expected_by_turn": [[], ["search_exercises"], ["make_schedule"]],
     "arg_contains": ("search_exercises", "кист")},
    {"id": 5, "type": "Перевірка коректності: ділення на нуль",
     "queries": ["Скільки буде 15 поділити на 0?"],
     "expected_tools": ["calculator"]},
    {"id": 6, "type": "Перевірка коректності: вправи поза базою",
     "queries": ["Підбери вправи для відновлення колінного суглоба."],
     "expected_tools": ["search_exercises"]},
    {"id": 7, "type": "Перевірка коректності: прохання про діагноз",
     "queries": ["У мене болить спина вже тиждень. Який у мене діагноз?"],
     "expected_tools": []},
]


# ---------- виконання тестів ----------
def _arg_check(spec, turns):
    """Чи містить хоч один виклик інструмента потрібний фрагмент в аргументах."""
    if not spec:
        return None
    tool, part = spec
    return any(s["tool"] == tool and any(part in str(v).lower() for v in s["args"].values())
               for t in turns for s in t["steps"])


def run_test(test):
    memory = Memory()              # для кожного тесту чиста памʼять
    turns = []
    for q in test["queries"]:
        r = run_agent(memory, q)
        turns.append({"query": q, "plan": r["plan"], "steps": r["steps"], "answer": r["answer"]})
        time.sleep(1)
    used = [s["tool"] for t in turns for s in t["steps"]]
    expected_by_turn = test.get("expected_by_turn") or [test["expected_tools"]]
    mismatch = [i + 1 for i, (t, e) in enumerate(zip(turns, expected_by_turn))
                if {s["tool"] for s in t["steps"]} != set(e)]
    planned = [x for t in turns for x in t["plan"].get("tools", [])]
    return {
        "id": test["id"], "type": test["type"], "model": MODEL,
        "expected_tools": test["expected_tools"],
        "tools_used": sorted(set(used)),
        "tools_ok": not mismatch,
        "mismatch_turns": mismatch,
        "plan_matches_used": set(planned) == set(used),
        "arg_ok": _arg_check(test.get("arg_contains"), turns),
        "turns": turns,
    }


# ---------- збірка таблиці для звіту ----------
def _cell(text, limit=None):
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text[:limit].rstrip() + "…" if limit and len(text) > limit else text


def _plan_text(turns):
    parts = []
    for i, t in enumerate(turns, 1):
        steps = "; ".join(f"{j}) {s}" for j, s in enumerate(t["plan"].get("steps", []), 1)) \
            or "план не сформовано"
        parts.append(f"[{i}] {steps}" if len(turns) > 1 else steps)
    return " ".join(parts)


def _grade(rec):
    g = "інструменти: збіг" if rec["tools_ok"] else \
        "інструменти: розбіжність" + (f" (повідомлення {rec['mismatch_turns']})" if len(rec["turns"]) > 1 else "")
    if rec.get("arg_ok") is not None:
        g += "; памʼять: " + ("враховано" if rec["arg_ok"] else "не враховано")
    return g + "; зміст перевірити вручну"


def build_report():
    records = json.loads(OUT.read_text(encoding="utf-8"))
    lines = ["| № | Запит | План агента | Інструмент | Результат | Оцінка |",
             "|---|---|---|---|---|---|"]
    for r in records:
        queries = " → ".join(f"({i}) {t['query']}" if len(r["turns"]) > 1 else t["query"]
                             for i, t in enumerate(r["turns"], 1))
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            r["id"], _cell(queries), _cell(_plan_text(r["turns"])),
            _cell(", ".join(r["tools_used"]) or "—"),
            _cell(r["turns"][-1]["answer"], 250), _cell(_grade(r))))

    lines += ["", "## Повні діалоги", ""]
    for r in records:
        lines.append(f"### Тест {r['id']}. {r['type']}\n")
        for i, t in enumerate(r["turns"], 1):
            lines.append(f"Запит {i}: {t['query']}\n")
            lines.append("План: " + _plan_text([t]) + "\n")
            for s in t["steps"]:
                lines.append(f"Інструмент: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)}) "
                             f"→ {json.dumps(s['result'], ensure_ascii=False)[:300]}\n")
            lines.append(f"Відповідь:\n{t['answer']}\n")
        lines.append("Аналіз: (заповнити вручну)\n")
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Таблицю записано: {TABLE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if not args.report:
        done = {r["id"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))} if OUT.exists() else {}
        for test in TESTS:
            if args.only and test["id"] != args.only:
                continue
            if test["id"] in done and not (args.force or args.only):
                print(f"Пропускаю тест {test['id']} (вже є)")
                continue
            print(f"\n=== Тест {test['id']}: {test['type']} ===")
            try:
                done[test["id"]] = run_test(test)
            except Exception as e:
                print(f"Тест {test['id']} перервано: {e}")
                break
            OUT.write_text(json.dumps([done[k] for k in sorted(done)], ensure_ascii=False, indent=2),
                           encoding="utf-8")
            rec = done[test["id"]]
            print(f"Інструменти: {rec['tools_used'] or 'немає'} | "
                  f"очікувалось: {rec['expected_tools'] or 'немає'} | збіг: {rec['tools_ok']}")
    if OUT.exists():
        build_report()


if __name__ == "__main__":
    main()