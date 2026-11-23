"""Завдання 2. Prompt Engineering: класифікація без додаткового контексту й прикладів."""
from main import ask
from utils import CATEGORIES, load_test_set, normalize

PROMPT_TEMPLATE = """Ти — фахівець з ерготерапії.
Визнач, до якого напряму ерготерапевтичного втручання належить наведена ситуація.
Можливі категорії: {categories}.
Поверни лише назву категорії.

Ситуація: {text}"""


def build_prompt(text):
    return PROMPT_TEMPLATE.format(categories=", ".join(CATEGORIES), text=text)


def main():
    correct = 0
    tests = load_test_set()
    for t in tests:
        answer = ask("task2", f"Prompt-{t['id']}", build_prompt(t["text"]), reuse=True)
        ok = normalize(answer) == t["correct"]
        correct += ok
        print(f"#{t['id']}: еталон={t['correct']} | модель={normalize(answer)} | {'✔' if ok else '✘'}")
    print(f"\nPrompt: {correct}/{len(tests)} = {correct / len(tests) * 100:.1f}%")


if __name__ == "__main__":
    main()