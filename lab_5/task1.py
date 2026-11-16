"""Завдання 1-2. Завантаження документів і розбиття на фрагменти (chunks).

Читає data/txt/*.txt, ділить кожен документ на chunks за реченнями
(розмір ~CHUNK_SIZE символів, перекриття ~OVERLAP символів між сусідніми chunks),
зберігає для кожного chunk його джерело у results/task1.json.
Ключ Gemini API для цього кроку не потрібен.
"""
import json
import re
from pathlib import Path

DATA_DIR = Path("data")
TXT_DIR = DATA_DIR / "txt"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 800   # бажана максимальна довжина chunk, символів
OVERLAP = 150      # скільки символів попереднього chunk повторюється на початку наступного

SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[А-ЯІЇЄҐA-Z«"(\-])')


def split_into_units(text):
    """Ділить текст на найменші одиниці: рядки (пункти списків, заголовки), а довгі рядки ще й на речення."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        units.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())
    return units


def hard_split(unit, size):
    """Запасний варіант для надто довгого речення: ділить за словами."""
    parts, current = [], ""
    for word in unit.split():
        if current and len(current) + 1 + len(word) > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def split_into_chunks(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Збирає речення в chunks довжиною до `size` символів із перекриттям `overlap`."""
    units = []
    for u in split_into_units(text):
        units.extend(hard_split(u, size) if len(u) > size else [u])

    chunks, current = [], []
    for u in units:
        length = sum(len(x) + 1 for x in current)
        if current and length + len(u) > size:
            chunks.append(" ".join(current))
            tail, total = [], 0
            for x in reversed(current):
                if total + len(x) > overlap:
                    break
                tail.insert(0, x)
                total += len(x) + 1
            current = tail
        current.append(u)
    if current:
        chunks.append(" ".join(current))
    return chunks


def find_original_name(stem):
    """document_1 -> document_1.pdf (оригінальна назва файлу з data/)."""
    for p in sorted(DATA_DIR.glob(f"{stem}.*")):
        if p.suffix.lower() != ".txt":
            return p.name
    return f"{stem}.txt"


def build_chunks():
    records, counter = [], 1
    for txt in sorted(TXT_DIR.glob("*.txt")):
        text = txt.read_text(encoding="utf-8")
        source = find_original_name(txt.stem)
        for i, chunk in enumerate(split_into_chunks(text), start=1):
            records.append({
                "chunk_id": f"chunk_{counter:03d}",
                "source": source,
                "chunk_in_doc": i,
                "n_chars": len(chunk),
                "text": chunk,
            })
            counter += 1
    return records


def print_structure(records):
    """Виводить структуру «документ -> chunks» (як у методичці)."""
    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)
    for source, items in by_source.items():
        print(f"\n{source}  ({len(items)} chunks)")
        for k, r in enumerate(items):
            branch = "└──" if k == len(items) - 1 else "├──"
            preview = r["text"][:60].replace("\n", " ")
            print(f"  {branch} {r['chunk_id']} [{r['n_chars']} симв.] {preview}...")


if __name__ == "__main__":
    records = build_chunks()
    (RESULTS_DIR / "task1.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print_structure(records)
    sizes = [r["n_chars"] for r in records]
    print(f"\nУсього chunks: {len(records)}")
    print(f"Довжина: мін. {min(sizes)}, середня {sum(sizes) // len(sizes)}, макс. {max(sizes)}")
    print("Збережено: results/task1.json")