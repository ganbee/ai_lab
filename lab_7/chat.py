"""Консольний чат з агентом.

Команди: /clear — очистити памʼять, /memory — кількість повідомлень у памʼяті, exit — вихід.
Діалог зберігається в results/chat_log.json.
"""
import json

from agent import RESULTS_DIR, Memory, run_agent


def main():
    memory = Memory()
    print("Агент з ерготерапії. Введіть запит (exit — вихід, /clear, /memory).\n")
    try:
        while True:
            try:
                text = input("Ви: ").strip()
            except EOFError:
                break
            if text.lower() in {"exit", "quit", "вихід"}:
                break
            if not text:
                continue
            if text == "/clear":
                memory.clear()
                print("Памʼять очищено.\n")
                continue
            if text == "/memory":
                print(f"У памʼяті {len(memory)} повідомлень.\n")
                continue

            result = run_agent(memory, text)

            plan = result["plan"]
            if plan.get("error"):
                print(f"  [план] не вдалося сформувати: {plan['error']}")
            else:
                for i, s in enumerate(plan["steps"], 1):
                    print(f"  [план] {i}. {s}")
                print(f"  [план] інструменти: {plan['tools'] or 'не потрібні'}")
            for s in result["steps"]:
                print(f"  [інструмент] {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})"
                      f" -> ok={s['result'].get('ok')}")
            print(f"\nАгент: {result['answer']}\n")
    finally:
        memory.save(RESULTS_DIR / "chat_log.json")


if __name__ == "__main__":
    main()