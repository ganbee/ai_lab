"""Дослідження top-p на іншій моделі (temperature = 1.0).
Використання: python task6_topp.py gemini-2.5-flash
Результат: results/task6_topp.json. Готові кроки пропускаються (reuse)."""
import sys
import main
from main import ask

model_id = sys.argv[1] if len(sys.argv) > 1 else "gemini-4-hydrogen"
main.MODEL = model_id
print("Модель:", model_id)

LOG = "task6_topp"
RUNS = 2
PROMPT = ("Запропонуй 5 ідей функціональних вправ для відновлення рівноваги "
          "у літніх пацієнтів після падіння. Для кожної вкажи назву та "
          "одне речення з описом.")

for p in (0.5, 0.8, 1.0):
    for r in range(1, RUNS + 1):
        ask(LOG, f"top_p={p} | прогін {r}", PROMPT,
            temperature=1.0, top_p=p, pause=4, reuse=True)