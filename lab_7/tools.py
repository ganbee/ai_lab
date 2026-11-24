"""Інструменти агента: calculator, search_exercises, make_schedule.

Кожен інструмент повертає словник {"ok": True/False, ...}, щоб модель бачила і помилки.
TOOL_DECLARATIONS — описи для Gemini; TOOL_FUNCTIONS — реальні Python-функції.
"""
import ast
import json
import math
import operator
import statistics
from datetime import date, timedelta
from pathlib import Path

EXERCISES_FILE = Path(__file__).parent / "data" / "exercises.json"

# ---------- 1. calculator ----------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"mean": lambda *a: statistics.fmean(a), "min": min, "max": max,
          "abs": abs, "round": round, "sqrt": math.sqrt}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("занадто великий степінь")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("недопустимий елемент виразу")


def calculator(expression: str) -> dict:
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return {"ok": False, "error": "Ділення на нуль"}
    except Exception as e:
        return {"ok": False, "error": f"Не вдалося обчислити: {e}"}
    if isinstance(result, float):
        result = round(result, 4)
    return {"ok": True, "expression": expression, "result": result}


# ---------- 2. search_exercises ----------
def _load_exercises():
    return json.loads(EXERCISES_FILE.read_text(encoding="utf-8"))


def _match(query, value):
    if not query:
        return True
    q, v = query.lower().strip(), value.lower()
    return q in v or v in q


def search_exercises(body_part: str = None, goal: str = None,
                     level: str = None, limit: int = 5) -> dict:
    items = _load_exercises()
    found = [e for e in items
             if _match(body_part, e["body_part"])
             and _match(goal, e["goal"])
             and _match(level, e["level"])]
    result = {"ok": True, "count": len(found), "exercises": found[:limit]}
    if not found:
        result["available"] = {
            "body_part": sorted({e["body_part"] for e in items}),
            "goal": sorted({e["goal"] for e in items}),
            "level": sorted({e["level"] for e in items}),
        }
        result["note"] = "Нічого не знайдено. Спробуй значення з поля available."
    return result


# ---------- 3. make_schedule ----------
_WEEKDAYS = ["понеділок", "вівторок", "середа", "четвер", "п’ятниця", "субота", "неділя"]
_PATTERNS = {1: [0], 2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4],
             5: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5, 6]}


def make_schedule(start_date: str, weeks: int, sessions_per_week: int) -> dict:
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        return {"ok": False, "error": "Дата має бути у форматі YYYY-MM-DD"}
    if not 1 <= weeks <= 12:
        return {"ok": False, "error": "Кількість тижнів: від 1 до 12"}
    if sessions_per_week not in _PATTERNS:
        return {"ok": False, "error": "Занять на тиждень: від 1 до 7"}
    pattern, total = _PATTERNS[sessions_per_week], weeks * sessions_per_week
    sessions, day = [], start
    while len(sessions) < total:
        if day.weekday() in pattern:
            sessions.append({"date": day.isoformat(), "weekday": _WEEKDAYS[day.weekday()]})
        day += timedelta(days=1)
    result = {"ok": True, "total": total, "sessions": sessions}
    if sessions[0]["date"] != start.isoformat():
        result["note"] = (f"Дата початку {start.isoformat()} ({_WEEKDAYS[start.weekday()]}) не входить у розклад, "
                          f"тому перше заняття припадає на {sessions[0]['date']} ({sessions[0]['weekday']}).")
    return result


# ---------- реєстр і описи для Gemini ----------
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "search_exercises": search_exercises,
    "make_schedule": make_schedule,
}

TOOL_DECLARATIONS = [
    {
        "name": "calculator",
        "description": "Виконує математичні обчислення. Використовуй для середніх значень, "
                       "відсотків, динаміки показників. Підтримує + - * / ** %, дужки та "
                       "функції mean(...), min(...), max(...), abs(...), round(...), sqrt(...).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "expression": {"type": "STRING",
                               "description": "Математичний вираз, наприклад (80+85+92+97)/4"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_exercises",
        "description": "Шукає вправи в локальній базі за ділянкою тіла, ціллю та рівнем. "
                       "Усі параметри необов'язкові. Повертає список вправ з описом і протипоказаннями.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "body_part": {"type": "STRING",
                              "description": "Ділянка: кисть, зап'ястя, лікоть, плече, обидві руки, тулуб"},
                "goal": {"type": "STRING",
                         "description": "Ціль: дрібна моторика, сила хвату, амплітуда рухів, "
                                        "самообслуговування, координація, баланс"},
                "level": {"type": "STRING",
                          "description": "Рівень: початковий, середній, просунутий"},
                "limit": {"type": "INTEGER", "description": "Максимум вправ у відповіді (за замовчуванням 5)"},
            },
        },
    },
    {
        "name": "make_schedule",
        "description": "Складає графік занять: список дат із днями тижня. "
                       "Використовуй, коли користувач просить розклад або план занять за датами.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "start_date": {"type": "STRING", "description": "Дата початку у форматі YYYY-MM-DD"},
                "weeks": {"type": "INTEGER", "description": "Кількість тижнів (1-12)"},
                "sessions_per_week": {"type": "INTEGER", "description": "Занять на тиждень (1-7)"},
            },
            "required": ["start_date", "weeks", "sessions_per_week"],
        },
    },
]


def run_tool(name: str, args: dict) -> dict:
    """Виконує інструмент за назвою. Помилки повертає як словник, а не кидає."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"ok": False, "error": f"Невідомий інструмент: {name}"}
    try:
        return func(**args)
    except TypeError as e:
        return {"ok": False, "error": f"Неправильні аргументи: {e}"}