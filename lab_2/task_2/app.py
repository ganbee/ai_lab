from taipy.gui import Gui, notify
import numpy as np

from functions import (
    process_image_for_lab2,
    train_hopfield,
    classify_hopfield,
    train_hamming,
    classify_hamming,
)

TEMP_A = "temp_grid_a2.png"
TEMP_B = "temp_grid_b2.png"
TEMP_C = "temp_grid_c2.png"
TEMP_TEST = "temp_grid_test2.png"

img_a = None
img_b = None
img_c = None
img_test = None

grid_a = None
grid_b = None
grid_c = None
grid_test = None

norm_a, norm_b, norm_c = None, None, None
bin_a, bin_b, bin_c = None, None, None

vec_a_str = "[]"
vec_b_str = "[]"
vec_c_str = "[]"
vec_test_str = "[]"

hopfield_weights_str = "Матриця Хопфілда не розрахована"
hamming_weights_str = "Ваги Хеммінга не розраховані"

hopfield_res = "Не визначено"
hamming_res = "Не визначено"
analysis_log = "Очікування введення даних...\n"

w_hopfield = None
w_hamming = None
b_hamming = None


def load_3_classes(state):
    if not state.img_a or not state.img_b or not state.img_c:
        notify(state, "warning", "Будь ласка, завантажте зображення для всіх 3-х класів!")
        return

    abs_a, norm_a, bin_a, p_a = process_image_for_lab2(state.img_a, TEMP_A)
    abs_b, norm_b, bin_b, p_b = process_image_for_lab2(state.img_b, TEMP_B)
    abs_c, norm_c, bin_c, p_c = process_image_for_lab2(state.img_c, TEMP_C)

    if norm_a and norm_b and norm_c:
        state.norm_a, state.norm_b, state.norm_c = norm_a, norm_b, norm_c
        state.bin_a, state.bin_b, state.bin_c = bin_a, bin_b, bin_c
        state.grid_a, state.grid_b, state.grid_c = p_a, p_b, p_c

        state.vec_a_str = f"Norm: {norm_a}\nBin: {bin_a}"
        state.vec_b_str = f"Norm: {norm_b}\nBin: {bin_b}"
        state.vec_c_str = f"Norm: {norm_c}\nBin: {bin_c}"

        state.w_hopfield = train_hopfield([bin_a, bin_b, bin_c])
        state.hopfield_weights_str = str(np.round(state.w_hopfield, 2).tolist())

        state.w_hamming, state.b_hamming = train_hamming([norm_a, norm_b, norm_c])
        state.hamming_weights_str = f"W1:\n{np.round(state.w_hamming, 4).tolist()}\n\nb1:\n{np.round(state.b_hamming, 2).tolist()}"

        notify(state, "success", "Еталони та матриці ваг успішно обчислено!")


def classify_all(state):
    if not state.img_test:
        notify(state, "error", "Завантажте тестове зображення!")
        return

    if state.w_hopfield is None or state.w_hamming is None:
        notify(state, "error", "Спочатку обробіть 3 еталонні класи!")
        return

    abs_test, norm_test, bin_test, p_test = process_image_for_lab2(state.img_test, TEMP_TEST)
    if not norm_test:
        return

    state.grid_test = p_test
    state.vec_test_str = f"Norm: {norm_test}\nBin: {bin_test}"

    class_names = ["Клас 1 (А)", "Клас 2 (B)", "Клас 3 (C)"]

    # Мережа Хопфілда
    hop_out, hop_iters = classify_hopfield(bin_test, state.w_hopfield)
    
    hop_arr = np.array(hop_out)
    dist_a = int(np.sum(hop_arr != np.array(state.bin_a)))
    dist_b = int(np.sum(hop_arr != np.array(state.bin_b)))
    dist_c = int(np.sum(hop_arr != np.array(state.bin_c)))
    
    dists = [dist_a, dist_b, dist_c]
    
    # якщо відстані унікальні, беремо мінімальну
    min_dist = min(dists)
    min_indices = [i for i, d in enumerate(dists) if d == min_dist]
    
    if len(min_indices) == 1:
        winner_hop = min_indices[0]
        state.hopfield_res = f"{class_names[winner_hop]} [Ітерацій: {hop_iters}]"
    else:
        # при однаковій відстані Хопфілда порівнюємо скалярний добуток з оригінальним бінарним тестом
        dot_a = int(np.dot(bin_test, state.bin_a))
        dot_b = int(np.dot(bin_test, state.bin_b))
        dot_c = int(np.dot(bin_test, state.bin_c))
        dots = [dot_a, dot_b, dot_c]
        winner_hop = int(np.argmax(dots))
        state.hopfield_res = f"{class_names[winner_hop]} [Ітерацій: {hop_iters}]"

    # Мережа Хеммінга
    ham_winner, ham_y2, ham_iters = classify_hamming(norm_test, state.w_hamming, state.b_hamming)
    state.hamming_res = f"{class_names[ham_winner]} [Ітерацій MAXNET: {ham_iters}]"

    state.analysis_log = (
        f"--- Результати аналізу ---\n"
        f"1. Мережа Хопфілда:\n"
        f"   - Відновлено стан за {hop_iters} ітерацій.\n"
        f"   - Відстані Хеммінга: A={dist_a}, E={dist_b}, Б={dist_c}\n"
        f"   - Результат: {state.hopfield_res}\n\n"
        f"2. Мережа Хеммінга:\n"
        f"   - MAXNET стабілізувався за {ham_iters} ітерацій.\n"
        f"   - Виходи 2-го шару: {ham_y2}\n"
        f"   - Переможець: {state.hamming_res}\n"
    )

    notify(state, "success", "Класифікацію виконано!")

page = """
# Лабораторна робота №2 — Частина 2: Мережі Хопфілда та Хеммінга

<|layout|columns=1 1 1|
<|
### Клас 1 (А)
<|{img_a}|file_selector|label=Обрати BMP Клас 1|extensions=.bmp|>
<br/>
<|{grid_a}|image|width=150px|>
<|{vec_a_str}|input|multiline=True|class_name=fullwidth|>
|>

<|
### Клас 2 (Е)
<|{img_b}|file_selector|label=Обрати BMP Клас 2|extensions=.bmp|>
<br/>
<|{grid_b}|image|width=150px|>
<|{vec_b_str}|input|multiline=True|class_name=fullwidth|>
|>

<|
### Клас 3 (В)
<|{img_c}|file_selector|label=Обрати BMP Клас 3|extensions=.bmp|>
<br/>
<|{grid_c}|image|width=150px|>
<|{vec_c_str}|input|multiline=True|class_name=fullwidth|>
|>
|>

<br/>
<|Обробити 3 еталони та обчислити ваги|button|on_action=load_3_classes|>

---

<|layout|columns=1 1|
<|
### Матриця ваг Хопфілда W (25x25)
<|{hopfield_weights_str}|input|multiline=True|class_name=fullwidth|>
|>

<|
### Вагові коефіцієнти Хеммінга
<|{hamming_weights_str}|input|multiline=True|class_name=fullwidth|>
|>
|>

---

## Класифікація тестового образу

<|layout|columns=1 1|
<|
### Невідомий (Тестовий) образ
<|{img_test}|file_selector|label=Обрати тестовий BMP|extensions=.bmp|>
<br/>
<|{grid_test}|image|width=180px|>

<br/>
<|Класифікувати (Хопфілд & Хеммінг)|button|on_action=classify_all|>

**Вектор ознак тестового образу:**
<|{vec_test_str}|input|multiline=True|class_name=fullwidth|>
|>

<|
### Результати класифікації

**Результат Мережі Хопфілда:**
### <|{hopfield_res}|text|>

**Результат Мережі Хеммінга:**
### <|{hamming_res}|text|>

**Аналіз процесу:**
<|{analysis_log}|input|multiline=True|class_name=fullwidth|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Мережі Хопфілда та Хеммінга", port=5554)