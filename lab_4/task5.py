import json
import sys
from main import ask, read_text, RESULTS_DIR

LOG = "task5"
text = read_text("text2.txt")

CHUNK_SIZE = 6000   
OVERLAP = 1000      

SYSTEM = ("Ти викладач фізичної терапії. Працюй лише з наданим текстом, "
          "нічого не вигадуй. Відповідай українською.")

FINAL_TASK = ("Склади узагальнення документа: 1) основна тема; 2) 7-10 ключових "
              "ідей; 3) основні поняття; 4) висновок. Структуруй відповідь "
              "заголовками.")


def make_chunks(text, size, overlap=0):
    """Ділить текст на фрагменти ~size символів по межі абзацу/речення.
    overlap > 0 дає sliding window (сусідні фрагменти частково перекриваються)."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def summarize_by_chunks(chunks, tag):
    summaries = []
    for i, ch in enumerate(chunks, 1):
        s = ask(LOG, f"{tag}. Chunk {i}/{len(chunks)}",
                f"Узагальни фрагмент документа (фрагмент {i} із {len(chunks)}) "
                f"у 4-6 реченнях: збережи ключові факти, терміни та числа."
                f"\n\nФрагмент:\n{ch}",
                system=SYSTEM, reuse=True, pause=15)
        summaries.append(f"[Фрагмент {i}]\n{s}")
    joined = "\n\n".join(summaries)
    return ask(LOG, f"{tag}. Загальне узагальнення",
               f"Нижче подано узагальнення послідовних фрагментів одного документа. "
               f"{FINAL_TASK}\n\nУзагальнення фрагментів:\n{joined}",
               system=SYSTEM, reuse=True, pause=15)


stats = {"chars": len(text), "words": len(text.split())}

ask(LOG, "5.1 Без chunking",
    f"{FINAL_TASK}\n\nДокумент:\n{text}", system=SYSTEM, reuse=True)

chunks = make_chunks(text, CHUNK_SIZE)
stats["chunks_simple"] = [len(c) for c in chunks]
summarize_by_chunks(chunks, "5.2 Chunking")

if "window" in sys.argv:
    windows = make_chunks(text, CHUNK_SIZE, OVERLAP)
    stats["chunks_sliding"] = [len(c) for c in windows]
    summarize_by_chunks(windows, "5.3 Sliding window")

(RESULTS_DIR / "task5_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print("Статистика:", stats)