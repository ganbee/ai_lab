"""Навчальний AI-агент з ерготерапії на Gemini.

Розділи файлу:
  1. Конфігурація
  2. Виклик моделі (LLM)
  3. Памʼять (Memory)
  4. Планувальник
  5. Головний цикл агента
"""
import json
import os
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import TOOL_DECLARATIONS, run_tool

# 1. КОНФІГУРАЦІЯ
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "gemini-3.5-flash-lite"   # 3.8 3.5-lite 4-hydrogen
TIMEOUT_MS = 120_000
TEMPERATURE = 0.0            
MAX_TOOL_STEPS = 6          
MAX_HISTORY = 20           

SYSTEM_PROMPT_TEMPLATE = """Ти — навчальний асистент з ерготерапії для студентів і початківців.

Правила:
1. Відповідай українською, без русизмів. Перевіряй написання термінів.
2. Не починай з вступних фраз («Ось...», «Звісно...»). Не використовуй емодзі та жирне форматування, якщо про це не просили. Десяткові дроби записуй з комою (88,5).
3. Усі обчислення виконуй лише через інструмент calculator. Не рахуй самостійно.
4. Вправи пропонуй лише ті, які повернув search_exercises. Якщо інструмент нічого не знайшов, прямо скажи про це. Не вигадуй вправ.
5. Графіки занять складай лише через make_schedule. Сьогоднішня дата: {today}. Якщо інструмент повернув поле note, коротко передай його користувачу.
6. Не став діагнозів і не призначай лікування. Для конкретного пацієнта рекомендуй консультацію лікаря або ерготерапевта.
7. Не вигадуй власного досвіду практики та посилань на дослідження.
8. Якщо запит не стосується ерготерапії чи реабілітації, коротко скажи, що це поза твоїми можливостями.
9. Якщо користувач лише повідомляє інформацію про себе чи пацієнта і нічого не просить, не викликай інструментів: коротко підтверди, що врахував її, і запитай, чим допомогти.
10. У search_exercises передавай лише ті параметри, які користувач назвав або які прямо випливають з діалогу. Не вигадуй рівень чи ціль.
"""


def get_system_prompt():
    return SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())


# 2. ВИКЛИК МОДЕЛІ (LLM)
load_dotenv(override=True)

_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=TIMEOUT_MS),
)


def make_tool(declarations):
    """Перетворює список словників-описів інструментів на types.Tool."""
    return types.Tool(
        function_declarations=[types.FunctionDeclaration(**d) for d in declarations]
    )


def generate(contents, system=None, tools=None, temperature=TEMPERATURE,
             json_mode=False, max_retries=5):
    """Один виклик моделі. contents — рядок або список повідомлень.

    Автоматичний виклик функцій вимкнено: цикл агента керується вручну.
    Повторюємо лише 429, 5xx і мережеві збої; 400 (помилка запиту) піднімається одразу.
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        tools=[make_tool(tools)] if tools else None,
        response_mime_type="application/json" if json_mode else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for attempt in range(max_retries):
        try:
            return _client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except errors.APIError as e:
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise RuntimeError("Вичерпано ДОБОВУ квоту. Повторіть запуск пізніше.")
            if e.code != 429 and e.code < 500:
                raise  # помилка запиту, повтор не допоможе
            print(f"Спроба {attempt + 1} невдала ({e.code}): {msg[:150]}")
        except Exception as e:  # таймаути, обрив зʼєднання
            print(f"Спроба {attempt + 1} невдала: {str(e)[:150]}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Не вдалося отримати відповідь після {max_retries} спроб")


def get_calls(response):
    """Список викликів інструментів (function_call), яких просить модель."""
    return list(response.function_calls or [])


def get_text(response):
    """Текст відповіді (порожній рядок, якщо модель повернула лише function_call)."""
    if not response.candidates or not response.candidates[0].content:
        return ""
    parts = response.candidates[0].content.parts or []
    return "".join(p.text for p in parts if getattr(p, "text", None)).strip()


# 3. ПАМʼЯТЬ (Memory)
def _has_text(content):
    return any(getattr(p, "text", None) for p in (content.parts or []))


class Memory:
    """Історія діалогу (types.Content) разом з викликами та результатами інструментів."""

    def __init__(self, max_messages=MAX_HISTORY):
        self.max_messages = max_messages
        self.contents = []

    def add(self, content):
        self.contents.append(content)

    def add_user_text(self, text):
        self.add(types.Content(role="user", parts=[types.Part(text=text)]))

    def window(self):
        """Останні max_messages повідомлень для передачі в модель.

        Вікно завжди починається зі звичайного повідомлення користувача (з текстом),
        щоб не обрізати пару «виклик інструмента — його результат».
        """
        items = self.contents[-self.max_messages:]
        for i, c in enumerate(items):
            if c.role == "user" and _has_text(c):
                return items[i:]
        return items

    def clear(self):
        self.contents = []

    def __len__(self):
        return len(self.contents)

    def transcript(self):
        """Спрощений запис діалогу для логів і звіту."""
        rows = []
        for c in self.contents:
            for p in c.parts or []:
                if getattr(p, "thought", None):
                    continue
                if getattr(p, "function_call", None):
                    rows.append({"role": "tool_call", "name": p.function_call.name,
                                 "args": dict(p.function_call.args or {})})
                elif getattr(p, "function_response", None):
                    rows.append({"role": "tool_result", "name": p.function_response.name,
                                 "result": p.function_response.response})
                elif getattr(p, "text", None):
                    rows.append({"role": "user" if c.role == "user" else "agent",
                                 "text": p.text})
        return rows

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.transcript(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")


# 4. ПЛАНУВАЛЬНИК
_TOOL_NAMES = {d["name"] for d in TOOL_DECLARATIONS}
_TOOLS_DESC = "\n".join(f"- {d['name']}: {d['description']}" for d in TOOL_DECLARATIONS)

PLANNER_PROMPT = (
    "Ти — модуль планування навчального агента з ерготерапії. "
    "Агент має такі інструменти:\n" + _TOOLS_DESC + "\n\n"
    "Склади короткий план виконання нового запиту користувача, враховуючи історію діалогу. "
    "Поверни лише JSON без пояснень у такому форматі:\n"
    '{"steps": ["крок 1", "крок 2"], "tools": ["назва_інструмента"]}\n\n'
    "Правила:\n"
    "- steps: від 1 до 5 коротких кроків українською, у наказовому способі.\n"
    "- tools: лише назви зі списку вище, які справді потрібні, у порядку використання. "
    "Якщо інструмент не потрібен, поверни порожній список.\n"
    "- Будь-які обчислення виконує тільки calculator. Вправи беруться тільки з search_exercises. "
    "Графіки за датами складає тільки make_schedule.\n"
    "- Якщо користувач лише повідомляє факти про себе чи пацієнта й нічого не просить, інструмент не потрібен: "
    "єдиний крок плану — підтвердити, що інформацію враховано, і запитати, чим допомогти.\n"
    "- Якщо запит просить діагноз чи лікування або не стосується ерготерапії, "
    "єдиний крок плану: ввічливо відмовити й пояснити межі агента."
)


def _context_text(contents, n=6):
    """Текстовий вигляд останніх повідомлень (без службових викликів інструментів)."""
    lines = []
    for content in contents[-n:]:
        text = " ".join(p.text for p in (content.parts or []) if getattr(p, "text", None))
        if text.strip():
            who = "Користувач" if content.role == "user" else "Агент"
            lines.append(f"{who}: {text.strip()[:400]}")
    return "\n".join(lines)


def _parse_plan(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(raw)
    steps = [str(s) for s in data.get("steps", [])][:5]
    tools = [t for t in data.get("tools", []) if t in _TOOL_NAMES]
    return {"steps": steps, "tools": tools, "needs_tools": bool(tools)}


def make_plan(contents, user_text):
    """Повертає {"steps", "tools", "needs_tools"}. Помилка планування не зупиняє агента."""
    context = _context_text(contents)
    prompt = (f"Історія діалогу:\n{context}\n\n" if context else "") + \
             f"Новий запит користувача:\n{user_text}"
    try:
        return _parse_plan(get_text(generate(prompt, system=PLANNER_PROMPT, json_mode=True)))
    except Exception as e:
        return {"steps": [], "tools": [], "needs_tools": None, "error": str(e)[:150]}


# 5. ГОЛОВНИЙ ЦИКЛ АГЕНТА
def _clean_args(args):
    """Gemini повертає числа як float (2.0). Перетворюємо цілі значення на int."""
    return {k: int(v) if isinstance(v, float) and v.is_integer() else v
            for k, v in dict(args or {}).items()}


def _plan_to_prompt(plan):
    if not plan["steps"]:
        return ""
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(plan["steps"], 1))
    return f"\n\nПлан дій для цього запиту (виконуй його послідовно):\n{steps}"


def run_agent(memory, user_text, use_planner=True):
    """Обробляє один запит користувача: план -> модель -> інструмент -> результат -> відповідь.

    memory — обʼєкт Memory (доповнюється на місці).
    Повертає {"answer", "plan", "steps": [{"tool", "args", "result"}, ...]}.
    """
    # планування виконується до додавання нового повідомлення в памʼять
    plan = make_plan(memory.contents, user_text) if use_planner else \
        {"steps": [], "tools": [], "needs_tools": None}

    memory.add_user_text(user_text)
    system = get_system_prompt() + _plan_to_prompt(plan)
    steps = []

    for _ in range(MAX_TOOL_STEPS):
        response = generate(memory.window(), system=system, tools=TOOL_DECLARATIONS)
        memory.add(response.candidates[0].content)

        calls = get_calls(response)
        if not calls:                        # інструмент не потрібен: фінальна відповідь
            return {"answer": get_text(response), "plan": plan, "steps": steps}

        result_parts = []
        for call in calls:
            args = _clean_args(call.args)
            result = run_tool(call.name, args)
            steps.append({"tool": call.name, "args": args, "result": result})
            result_parts.append(types.Part.from_function_response(
                name=call.name, response={"result": result}))
        memory.add(types.Content(role="user", parts=result_parts))

    answer = "Не вдалося завершити запит за допустиму кількість кроків."
    memory.add(types.Content(role="model", parts=[types.Part(text=answer)]))
    return {"answer": answer, "plan": plan, "steps": steps}
