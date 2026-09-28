from taipy.gui import Gui, notify
from PIL import Image, ImageDraw
import numpy as np
import os

# --- Константи ---
GRID_ROWS = 5
GRID_COLS = 5

# --- Змінні стану ---
ref_1_path = None
ref_2_path = None
ref_3_path = None
ref_4_path = None

ref_1_grid = None
ref_2_grid = None
ref_3_grid = None
ref_4_grid = None

ref_1_abs = ""
ref_1_norm = ""
ref_2_abs = ""
ref_2_norm = ""
ref_3_abs = ""
ref_3_norm = ""
ref_4_abs = ""
ref_4_norm = ""

unknown_path = None
unknown_grid = None
unknown_abs = ""
unknown_norm = ""

metric_choice = "Чебишевська норма"
distances_output = ""
classification_output = ""

# Збережені обчислені вектори
ref_vectors = [None, None, None, None]  # [(abs, norm), ...]
unknown_vector_data = None  # (abs, norm)


# --- Допоміжні функції ---
def calculate_feature_vector(image_path):
    if not image_path or not os.path.exists(image_path):
        return None, None

    image = Image.open(image_path).convert("L")
    width, height = image.size
    pixels = np.array(image)

    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS

    absolute_vector = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)
            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)

            cell = pixels[y1:y2, x1:x2]
            black_pixels = np.sum(cell < 128)
            absolute_vector.append(int(black_pixels))

    total_black_pixels = sum(absolute_vector)

    if total_black_pixels > 0:
        normalized_vector = [
            round(value / total_black_pixels, 4) for value in absolute_vector
        ]
    else:
        normalized_vector = [0.0] * len(absolute_vector)

    return absolute_vector, normalized_vector


def create_grid_image(image_path, output_filename):
    if not image_path or not os.path.exists(image_path):
        return None

    image = Image.open(image_path).convert("L")
    width, height = image.size

    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS

    grid_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(grid_image)

    for col in range(1, GRID_COLS):
        x = int(col * cell_width)
        draw.line((x, 0, x, height), fill="red", width=2)

    for row in range(1, GRID_ROWS):
        y = int(row * cell_height)
        draw.line((0, y, width, y), fill="red", width=2)

    grid_image.save(output_filename)
    return output_filename


# --- Метрики відстані ---
def calculate_distance(vec1, vec2, metric):
    if metric == "Евклідова норма":
        # d_1 = sqrt( sum( (x_k^1 - x_k^2)^2 ) )
        return float(np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2)))
    elif metric == "Чебишевська норма":
        # d_3 = max( |x_k^1 - x_k^2| )
        return float(np.max(np.abs(np.array(vec1) - np.array(vec2))))
    elif metric == "Манхетенська норма":
        # d_2 = sum( |x_k^1 - x_k^2| )
        return float(np.sum(np.abs(np.array(vec1) - np.array(vec2))))
    return 0.0


# --- Обробка подій (Actions) ---
def process_references(state):
    paths = [state.ref_1_path, state.ref_2_path, state.ref_3_path, state.ref_4_path]

    for i, path in enumerate(paths, start=1):
        if not path:
            notify(state, "error", f"Будь ласка, завантажте зображення для Класу {i}.")
            return

    for i, path in enumerate(paths):
        abs_vec, norm_vec = calculate_feature_vector(path)
        state.ref_vectors[i] = (abs_vec, norm_vec)
        grid_file = f"ref_{i+1}_grid.png"
        create_grid_image(path, grid_file)

        if i == 0:
            state.ref_1_abs, state.ref_1_norm, state.ref_1_grid = str(abs_vec), str(norm_vec), grid_file
        elif i == 1:
            state.ref_2_abs, state.ref_2_norm, state.ref_2_grid = str(abs_vec), str(norm_vec), grid_file
        elif i == 2:
            state.ref_3_abs, state.ref_3_norm, state.ref_3_grid = str(abs_vec), str(norm_vec), grid_file
        elif i == 3:
            state.ref_4_abs, state.ref_4_norm, state.ref_4_grid = str(abs_vec), str(norm_vec), grid_file

    notify(state, "success", "Усі еталонні образи успішно оброблено!")


def recognize_unknown(state):
    if not state.unknown_path:
        notify(state, "error", "Будь ласка, завантажте невідоме зображення.")
        return

    if any(v is None for v in state.ref_vectors):
        notify(state, "error", "Спочатку обробіть всі 4 еталонні образи!")
        return

    # Обчислюємо вектори невідомого образу
    abs_vec, norm_vec = calculate_feature_vector(state.unknown_path)
    grid_file = "unknown_grid.png"
    create_grid_image(state.unknown_path, grid_file)

    state.unknown_grid = grid_file
    state.unknown_abs = str(abs_vec)
    state.unknown_norm = str(norm_vec)

    # Обчислюємо відстані
    distances = []
    for i, (ref_abs, ref_norm) in enumerate(state.ref_vectors, start=1):
        dist = calculate_distance(norm_vec, ref_norm, state.metric_choice)
        distances.append(dist)

    min_dist = min(distances)
    nearest_class = distances.index(min_dist) + 1

    dist_text = f"Метрика: {state.metric_choice}\n\n"
    for i, d in enumerate(distances, start=1):
        dist_text += f"Класу {i}: {d:.4f}\n"

    state.distances_output = dist_text
    state.classification_output = (
        f"РЕЗУЛЬТАТ КЛАСИФІКАЦІЇ:\n\n"
        f"Невідомий образ належить до КЛАСУ {nearest_class}\n"
        f"Мінімальна відстань: {min_dist:.4f}"
    )

    notify(state, "success", "Розпізнавання завершено!")


# --- Інтерфейс Taipy ---
page = """
# Система розпізнавання образів за методом порівняння з еталоном

---

## 1. Еталонні образи (4 Класи)

<|layout|columns=1 1 1 1|
<|
### Клас 1
<|{ref_1_path}|file_selector|label=Завантажити Еталон 1|extensions=.bmp,.png,.jpg|>
<|{ref_1_grid}|image|width=180px|>
<|{ref_1_abs}|input|multiline=True|label=Абсолютний вектор|>
<|{ref_1_norm}|input|multiline=True|label=Нормований вектор|>
|>

<|
### Клас 2
<|{ref_2_path}|file_selector|label=Завантажити Еталон 2|extensions=.bmp,.png,.jpg|>
<|{ref_2_grid}|image|width=180px|>
<|{ref_2_abs}|input|multiline=True|label=Абсолютний вектор|>
<|{ref_2_norm}|input|multiline=True|label=Нормований вектор|>
|>

<|
### Клас 3
<|{ref_3_path}|file_selector|label=Завантажити Еталон 3|extensions=.bmp,.png,.jpg|>
<|{ref_3_grid}|image|width=180px|>
<|{ref_3_abs}|input|multiline=True|label=Абсолютний вектор|>
<|{ref_3_norm}|input|multiline=True|label=Нормований вектор|>
|>

<|
### Клас 4
<|{ref_4_path}|file_selector|label=Завантажити Еталон 4|extensions=.bmp,.png,.jpg|>
<|{ref_4_grid}|image|width=180px|>
<|{ref_4_abs}|input|multiline=True|label=Абсолютний вектор|>
<|{ref_4_norm}|input|multiline=True|label=Нормований вектор|>
|>
|>

<br/>
<|Обробити еталонні образи|button|on_action=process_references|>

---

## 2. Невідомий образ та розпізнавання

<|layout|columns=1 1 1|
<|
### Завантаження
<|{unknown_path}|file_selector|label=Завантажити образ|extensions=.bmp,.png,.jpg|>
<|{unknown_grid}|image|width=220px|>
<|{metric_choice}|selector|lov=Чебишевська норма;Евклідова норма;Манхетенська норма|dropdown=True|label=Виберіть норму (метрику)|>

<br/>
<|Розпізнати образ|button|on_action=recognize_unknown|>
|>

<|
### Вектори ознак
<|{unknown_abs}|input|multiline=True|label=Абсолютний вектор невідомого образу|>
<|{unknown_norm}|input|multiline=True|label=Нормований вектор невідомого образу|>
|>

<|
### Результати
<|{distances_output}|input|multiline=True|label=Міри відповідності (відстані)|>
<|{classification_output}|input|multiline=True|label=Висновок|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Розпізнавання образів (Task 2)", dark_mode=False, port=5552)