"""Завдання 4. Імітація fine-tuning: few-shot prompting (набір прикладів «текст → категорія»)."""
import json

from main import ask, DATA_DIR
from utils import CATEGORIES, load_test_set, normalize

INSTRUCTION = """Ти — фахівець з ерготерапії.
Визнач, до якого напряму ерготерапевтичного втручання належить наведена ситуація.
Можливі категорії: {categories}.
Поверни лише назву категорії.
Нижче наведено приклади правильної класифікації."""


def load_examples():
    return json.loads((DATA_DIR / "few_shot.json").read_text(encoding="utf-8"))


def build_prompt(text, examples):
    parts = [INSTRUCTION.format(categories=", ".join(CATEGORIES))]
    for ex in examples:
        parts.append(f"Текст:\n{ex['text']}\nКатегорія:\n{ex['category']}")
    parts.append(f"Текст:\n{text}\nКатегорія:")
    return "\n\n".join(parts)


def main():
    examples = load_examples()
    tests = load_test_set()
    correct = 0
    for t in tests:
        answer = ask("task4", f"FewShot-{t['id']}", build_prompt(t["text"], examples), reuse=True)
        ok = normalize(answer) == t["correct"]
        correct += ok
        print(f"#{t['id']}: еталон={t['correct']} | модель={normalize(answer)} | {'✔' if ok else '✘'}")
    print(f"\nFew-shot: {correct}/{len(tests)} = {correct / len(tests) * 100:.1f}%")


if __name__ == "__main__":
    main()