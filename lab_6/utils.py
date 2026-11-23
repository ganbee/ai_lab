import json
from main import DATA_DIR

CATEGORIES = [
    "повсякденна діяльність",
    "функція руки",
    "когнітивна реабілітація",
    "адаптація середовища",
]


def load_test_set():
    """Читає data/test_set.json."""
    path = DATA_DIR / "test_set.json"
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(answer):
    """Приводить відповідь моделі до однієї з категорій (або повертає очищений текст)."""
    text = (answer or "").strip().lower().strip(" .,:;\"'«»*`\n")
    for cat in CATEGORIES:
        if text == cat:
            return cat
    for cat in CATEGORIES:  # на випадок зайвих слів у відповіді
        if cat in text:
            return cat
    return text