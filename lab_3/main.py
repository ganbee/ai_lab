import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

MODEL = "gemini-3.6-flash"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60_000),  
)


def ask(log, label, prompt, system=None, temperature=0.0):
    """Надсилає запит, зберігає у results/<log>.json, повертає відповідь."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
    )

    print(f"Надсилаю: {label} ...")
    response = None
    for attempt in range(5):
        try:
            response = _client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            break
        except Exception as e:
            print(f"Спроба {attempt + 1} невдала: {str(e)[:200]}")
            time.sleep(15 * (attempt + 1))
    if response is None:
        raise RuntimeError("Не вдалося отримати відповідь після 5 спроб")
    answer = response.text

    path = RESULTS_DIR / f"{log}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    data = [d for d in data if d["label"] != label]
    data.append({
        "label": label,
        "model": response.model_version or MODEL,
        "system": system,
        "prompt": prompt,
        "answer": answer,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {label} ===\n{answer}\n")
    time.sleep(8)  # пауза між запитами
    return answer


if __name__ == "__main__":
    ask("test", "Тест", "Поясни, що таке LLM, у двох реченнях.")