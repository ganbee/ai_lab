import json
import re
from itertools import combinations
from main import ask, RESULTS_DIR

LOG = "task6"
RUNS = 2 

PROMPT = ("Запропонуй 5 ідей функціональних вправ для відновлення рівноваги "
          "у літніх пацієнтів після падіння. Для кожної вкажи назву та "
          "одне речення з описом.")


def words(s):
    return set(re.findall(r"\w+", s.lower()))


def similarity(answers):
    """Середня схожість (Jaccard) між прогонами: 1 = однакові, 0 = різні."""
    pairs = list(combinations(answers, 2))
    return sum(len(words(a) & words(b)) / len(words(a) | words(b))
               for a, b in pairs) / len(pairs)


metrics = []


def experiment(group, value, **params):
    answers = []
    for r in range(1, RUNS + 1):
        answers.append(ask(LOG, f"{group}={value} | прогін {r}", PROMPT,
                           pause=4, **params))
    metrics.append({
        "group": group, "value": value,
        "avg_similarity_between_runs": round(similarity(answers), 3),
        "avg_length_words": round(sum(len(a.split()) for a in answers) / RUNS, 1),
    })


for t in (0, 0.5, 1):
    experiment("temperature", t, temperature=t)

for p in (0.5, 0.8, 1.0):
    experiment("top_p", p, temperature=1.0, top_p=p)

(RESULTS_DIR / "task6_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
for m in metrics:
    print(m)