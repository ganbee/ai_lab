import json
import re
from main import ask, read_text, RESULTS_DIR

LOG = "task4"
text = read_text("text1.txt")

prompt = f"""Проаналізуй текст зі сфери фізичної терапії та поверни результат
ТІЛЬКИ у форматі JSON, без пояснень і без markdown-огорожі.

Структура:
{{
  "topic": "основна тема одним реченням",
  "key_ideas": ["3-5 ключових ідей"],
  "keywords": ["5-8 ключових слів"],
  "conclusion": "короткий висновок, до 2 речень"
}}

Текст:
{text}"""

raw_a = ask(LOG, "A. JSON лише через prompt", prompt)

raw_b = ask(LOG, "B. JSON через response_mime_type", prompt, json_mode=True)

SCHEMA = {"topic": str, "key_ideas": list, "keywords": list, "conclusion": str}


def check(label, raw):
    info = {"label": label, "valid_json_as_is": False, "valid_after_cleanup": False,
            "errors": []}
    data = None
    try:
        data = json.loads(raw)
        info["valid_json_as_is"] = True
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            data = json.loads(cleaned)
            info["valid_after_cleanup"] = True
        except json.JSONDecodeError as e:
            info["errors"].append(f"JSON не розібрано: {e}")

    if data is not None:
        for key, typ in SCHEMA.items():
            if key not in data:
                info["errors"].append(f"немає ключа '{key}'")
            elif not isinstance(data[key], typ):
                info["errors"].append(f"ключ '{key}' має тип {type(data[key]).__name__}")
        if isinstance(data.get("key_ideas"), list) and not 3 <= len(data["key_ideas"]) <= 5:
            info["errors"].append("key_ideas: має бути 3-5 елементів")
        if isinstance(data.get("keywords"), list) and not 5 <= len(data["keywords"]) <= 8:
            info["errors"].append("keywords: має бути 5-8 елементів")
        extra = set(data) - set(SCHEMA)
        if extra:
            info["errors"].append(f"зайві ключі: {sorted(extra)}")
        info["parsed"] = data

    info["structure_ok"] = data is not None and not info["errors"]
    return info


checks = [check("A. JSON лише через prompt", raw_a),
          check("B. JSON через response_mime_type", raw_b)]

for c in checks:
    print(f"\n[{c['label']}] чистий JSON: {c['valid_json_as_is']}, "
          f"після очищення: {c['valid_after_cleanup']}, "
          f"структура ОК: {c['structure_ok']}")
    for err in c["errors"]:
        print("  -", err)

(RESULTS_DIR / "task4_check.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")