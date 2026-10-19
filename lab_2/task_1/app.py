from taipy.gui import Gui, notify
import numpy as np

from functions import (
    N_FEATURES,
    process_image_and_save_grid,
    calculate_perceptron_output,
    update_weights,
)

TEMP_GRID_A = "temp_grid_a.png"
TEMP_GRID_B = "temp_grid_b.png"
TEMP_GRID_TEST = "temp_grid_test.png"

#змінні стану
img_class_a = None
img_class_b = None
img_test = None

grid_a = None
grid_b = None
grid_test = None

abs_vec_a_str = "[]"
norm_vec_a_str = "[]"
abs_vec_b_str = "[]"
norm_vec_b_str = "[]"

abs_vec_test_str = "[]"
norm_vec_test_str = "[]"

vec_a_data = None
vec_b_data = None
vec_test_data = None

# параметри перцептрона
learning_rate = 0.1
bias = 0.1
weights = [0.05] * N_FEATURES
weights_str = str(weights)

predicted_class = "Не визначено"
desired_class = 1  # +1 для Класу A, -1 для Класу B
training_log = "Історія навчання порожня.\n"
iteration_count = 0


def load_data(state):
    """Обробляє обидва еталони та будує абсолютні й нормовані вектори."""
    if not state.img_class_a or not state.img_class_b:
        notify(state, "warning", "Будь ласка, оберіть файли для Класів A та B!")
        return

    abs_a, norm_a, path_a = process_image_and_save_grid(state.img_class_a, TEMP_GRID_A)
    abs_b, norm_b, path_b = process_image_and_save_grid(state.img_class_b, TEMP_GRID_B)

    if norm_a and norm_b:
        state.vec_a_data = norm_a
        state.vec_b_data = norm_b

        state.abs_vec_a_str = str(abs_a)
        state.norm_vec_a_str = str(norm_a)

        state.abs_vec_b_str = str(abs_b)
        state.norm_vec_b_str = str(norm_b)

        state.grid_a = path_a
        state.grid_b = path_b

        notify(state, "success", "Абсолютні та нормовані вектори успішно побудовано!")


def predict(state):
    """Класифікація тестового образу."""
    if not state.img_test:
        notify(state, "error", "Завантажте тестовий образ!")
        return

    abs_test, norm_test, path_test = process_image_and_save_grid(state.img_test, TEMP_GRID_TEST)
    if not norm_test:
        notify(state, "error", "Не вдалося зчитати тестовий образ.")
        return

    state.vec_test_data = norm_test
    state.abs_vec_test_str = str(abs_test)
    state.norm_vec_test_str = str(norm_test)
    state.grid_test = path_test

    s, y = calculate_perceptron_output(norm_test, state.weights, float(state.bias))
    state.predicted_class = "Клас A (+1)" if y == 1 else "Клас B (-1)"

    notify(state, "success", f"Класифікація виконана (S={s}): {state.predicted_class}")


def on_confirm(state):
    """Підтвердження — ваги не змінюємо."""
    if not state.vec_test_data:
        notify(state, "error", "Спочатку проведіть класифікацію!")
        return

    state.iteration_count += 1
    s, y = calculate_perceptron_output(state.vec_test_data, state.weights, float(state.bias))
    log_msg = f"Крок {state.iteration_count}: ✓ [Правильно] Помилки немає (y={y}). Ваги не змінюються.\n"
    state.training_log += log_msg
    notify(state, "info", "Результат підтверджено.")


def on_refute(state):
    """Спростування — оновлення ваг відповідно до бажаного класу (d)."""
    if not state.vec_test_data:
        notify(state, "error", "Спочатку проведіть класифікацію!")
        return

    state.iteration_count += 1
    r = float(state.learning_rate)
    b = float(state.bias)
    d = int(state.desired_class)

    s, y = calculate_perceptron_output(state.vec_test_data, state.weights, b)

    if y == d:
        notify(state, "info", f"Вихід перцептрона вже відповідає бажаному класу d={d}.")
        return

    state.weights = update_weights(state.weights, state.vec_test_data, d, y, r)
    state.weights_str = str(state.weights)

    error = d - y
    log_msg = f"Крок {state.iteration_count}: ✗ [Корекція] Помилка e={error} (y={y}, d={d}). Ваги скориговано!\n"
    state.training_log += log_msg

    # перераховуємо новий вихід після корекції
    s_new, y_new = calculate_perceptron_output(state.vec_test_data, state.weights, b)
    state.predicted_class = "Клас A (+1)" if y_new == 1 else "Клас B (-1)"

    notify(state, "warning", f"Ваги оновлено! Новий вихід: {state.predicted_class}")


page = """
# Лабораторна робота №2 — Перцептрон Розенблата

<|layout|columns=1 1|
<|
### 1. Навчальна вибірка (Еталони)

<|layout|columns=1 1|
<|
**Клас A (+1):**
<|{img_class_a}|file_selector|label=Обрати BMP Клас A|extensions=.bmp|>
<br/>
<|{grid_a}|image|width=200px|>
|>

<|
**Клас B (-1):**
<|{img_class_b}|file_selector|label=Обрати BMP Клас B|extensions=.bmp|>
<br/>
<|{grid_b}|image|width=200px|>
|>
|>

<br/>
<|Обробити еталони|button|on_action=load_data|>

**Абсолютний вектор Класу A:**
<|{abs_vec_a_str}|input|multiline=True|class_name=fullwidth|>

**Нормований вектор Класу A:**
<|{norm_vec_a_str}|input|multiline=True|class_name=fullwidth|>

**Абсолютний вектор Класу B:**
<|{abs_vec_b_str}|input|multiline=True|class_name=fullwidth|>

**Нормований вектор Класу B:**
<|{norm_vec_b_str}|input|multiline=True|class_name=fullwidth|>
|>

<|
### 2. Параметри Перцептрона
**Швидкість навчання (r):**
<|{learning_rate}|number|>

**Зсув / Поріг (b):**
<|{bias}|number|>

**Поточні синаптичні ваги (W):**
<|{weights_str}|input|multiline=True|class_name=fullwidth|>
|>
|>

---

## 3. Класифікація та Навчання з учителем

<|layout|columns=1 1|
<|
### Тестовий образ
<|{img_test}|file_selector|label=Обрати тестовий BMP|extensions=.bmp|>

<br/>
<|{grid_test}|image|width=200px|>

<br/>
<|Класифікувати|button|on_action=predict|>

**Абсолютний вектор тестового образу:**
<|{abs_vec_test_str}|input|multiline=True|class_name=fullwidth|>

**Нормований вектор тестового образу:**
<|{norm_vec_test_str}|input|multiline=True|class_name=fullwidth|>

**Результат класифікації:**
### <|{predicted_class}|text|>
|>

<|
### Оцінка вчителя та Корекція

**Бажаний клас (d):**
<|{desired_class}|number|> *(введіть +1 для Класу A, -1 для Класу B)*

<br/>
<|Підтвердити (Правильно)|button|on_action=on_confirm|>
<|Спростувати (Корекція ваг)|button|on_action=on_refute|>

**Журнал навчання та збіжності:**
<|{training_log}|input|multiline=True|class_name=fullwidth|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Перцептрон Розенблата", port=5553)