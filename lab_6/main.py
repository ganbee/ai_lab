import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

MODEL = "gemini-3.8-flash"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # розмірність вектора 
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=120_000),
)


def read_text(name):
    """Читає текст із data/<name>."""
    return (DATA_DIR / name).read_text(encoding="utf-8").strip()


def ask(log, label, prompt, system=None, temperature=0.0, top_p=None,
        max_output_tokens=None, json_mode=False, pause=8, reuse=False):
    """Надсилає запит, зберігає у results/<log>.json, повертає відповідь.

    Якщо запис із таким label уже є, він перезаписується.
    """
    path = RESULTS_DIR / f"{log}.json"
    if reuse and path.exists():
        for d in json.loads(path.read_text(encoding="utf-8")):
            if d["label"] == label and d.get("answer"):
                print(f"Пропускаю (вже є): {label}")
                return d["answer"]

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if json_mode else None,
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
            msg = str(e)
            print(f"Спроба {attempt + 1} невдала: {msg[:200]}")
            if "PerDay" in msg or "per day" in msg.lower():
                raise RuntimeError(
                    "Вичерпано ДОБОВУ квоту. Повторіть запуск пізніше: "
                    "виконані кроки збережено й будуть пропущені.")
            time.sleep(15 * (attempt + 1))
    if response is None:
        raise RuntimeError("Не вдалося отримати відповідь після 5 спроб")
    answer = response.text

    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    data = [d for d in data if d["label"] != label]
    data.append({
        "label": label,
        "model": response.model_version or MODEL,
        "temperature": temperature,
        "top_p": top_p,
        "system": system,
        "prompt": prompt,
        "answer": answer,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {label} ===\n{answer}\n")
    time.sleep(pause)  # пауза між запитами
    return answer


def embed_texts(texts, task_type, batch_size=20, pause=4):
    """Повертає список embedding-векторів для списку текстів.

    task_type: "RETRIEVAL_DOCUMENT" для фрагментів бази знань,
               "RETRIEVAL_QUERY" для запитів користувача.
    Тексти надсилаються пакетами по batch_size, щоб не впиратися в ліміти API.
    """
    vectors = []
    config = types.EmbedContentConfig(
        task_type=task_type, output_dimensionality=EMBED_DIM
    )
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = None
        for attempt in range(5):
            try:
                response = _client.models.embed_content(
                    model=EMBED_MODEL, contents=batch, config=config
                )
                break
            except Exception as e:
                msg = str(e)
                print(f"Спроба {attempt + 1} невдала: {msg[:200]}")
                if "PerDay" in msg or "per day" in msg.lower():
                    raise RuntimeError(
                        "Вичерпано ДОБОВУ квоту. Повторіть запуск пізніше.")
                time.sleep(15 * (attempt + 1))
        if response is None:
            raise RuntimeError("Не вдалося отримати embeddings після 5 спроб")
        vectors.extend(emb.values for emb in response.embeddings)
        print(f"Embeddings: {len(vectors)}/{len(texts)}")
        if i + batch_size < len(texts):
            time.sleep(pause)
    return vectors


if __name__ == "__main__":
    ask("test", "Тест", "Поясни, що таке LLM, у двох реченнях.")