from taipy.gui import Gui, notify
from PIL import Image, ImageDraw
import numpy as np
import os

# --- Константи ---
GRID_ROWS = 5
GRID_COLS = 5

# --- Змінні стану: Шляхи до вихідних файлів ---
c1_p1 = None
c1_p2 = None
c1_p3 = None

c2_p1 = None
c2_p2 = None
c2_p3 = None

c3_p1 = None
c3_p2 = None
c3_p3 = None

# --- Змінні стану: Шляхи до згенерованих фото ІЗ СІТКОЮ ---
c1_g1 = None
c1_g2 = None
c1_g3 = None

c2_g1 = None
c2_g2 = None
c2_g3 = None

c3_g1 = None
c3_g2 = None
c3_g3 = None

# Текстові блоки результатів
c1_stats_text = ""
c2_stats_text = ""
c3_stats_text = ""

training_status = "Система ще не навчена."

# Внутрішні дані моделей
cluster_centers = []     # [[c1], [c2], [c3]]
cluster_min_max = []    # [[(min, max), ...], ...]

# Невідомий образ
unknown_path = None
unknown_grid = None
unknown_abs = ""
unknown_norm = ""

classification_method = "Геометричні центри (Евклідова відстань)"
classification_result = ""


# --- Логіка обчислень ---
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


def process_class_samples(paths, class_prefix):
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    if not valid_paths:
        return "Зразки не завантажені", None, None, []

    norm_vectors = []
    grid_paths = []

    for idx, path in enumerate(paths, start=1):
        if path and os.path.exists(path):
            _, norm_v = calculate_feature_vector(path)
            norm_vectors.append(norm_v)

            # Генерація зображення з сіткою
            grid_filename = f"train_{class_prefix}_{idx}_grid.png"
            create_grid_image(path, grid_filename)
            grid_paths.append(grid_filename)
        else:
            grid_paths.append(None)

    norm_matrix = np.array(norm_vectors)

    # 1. Геометричний центр (M[X])
    center = np.mean(norm_matrix, axis=0)
    center_rounded = [round(float(val), 4) for val in center]

    # 2. Min/Max значення компонент
    min_vals = np.min(norm_matrix, axis=0)
    max_vals = np.max(norm_matrix, axis=0)
    min_max_bounds = list(zip(min_vals, max_vals))

    stats_text = (
        f"Завантажено зразків: {len(norm_vectors)}\n\n"
        f"Геометричний центр (M[X]):\n{center_rounded}\n\n"
        f"Min межі:\n{[round(float(v), 4) for v in min_vals]}\n\n"
        f"Max межі:\n{[round(float(v), 4) for v in max_vals]}"
    )

    return stats_text, center_rounded, min_max_bounds, grid_paths


# --- Обробники подій (Actions) ---
def train_system(state):
    c1_paths = [state.c1_p1, state.c1_p2, state.c1_p3]
    c2_paths = [state.c2_p1, state.c2_p2, state.c2_p3]
    c3_paths = [state.c3_p1, state.c3_p2, state.c3_p3]

    classes_data = [
        (c1_paths, "c1", "КЛАС 1"),
        (c2_paths, "c2", "КЛАС 2"),
        (c3_paths, "c3", "КЛАС 3")
    ]

    centers = []
    min_maxes = []

    for idx, (paths, prefix, name) in enumerate(classes_data, start=1):
        s_text, center, min_max, grids = process_class_samples(paths, prefix)

        if center is None:
            notify(state, "error", f"Завантажте хоча б один зразок для {name}!")
            return

        centers.append(center)
        min_maxes.append(min_max)

        # Оновлення сіток та статистики в інтерфейсі
        if idx == 1:
            state.c1_stats_text = s_text
            state.c1_g1, state.c1_g2, state.c1_g3 = grids[0], grids[1], grids[2]
        elif idx == 2:
            state.c2_stats_text = s_text
            state.c2_g1, state.c2_g2, state.c2_g3 = grids[0], grids[1], grids[2]
        elif idx == 3:
            state.c3_stats_text = s_text
            state.c3_g1, state.c3_g2, state.c3_g3 = grids[0], grids[1], grids[2]

    state.cluster_centers = centers
    state.cluster_min_max = min_maxes
    state.training_status = "НАВЧАННЯ УСПІШНО ЗАВЕРШЕНО!\nУсі зображення оброблено сіткою 5x5."
    notify(state, "success", "Навчання системи завершено!")


def classify_unknown(state):
    if not state.unknown_path:
        notify(state, "error", "Завантажте невідомий образ!")
        return

    if not state.cluster_centers:
        notify(state, "error", "Спочатку виконайте навчання системи!")
        return

    abs_v, norm_v = calculate_feature_vector(state.unknown_path)
    grid_file = "unknown_task3_grid.png"
    create_grid_image(state.unknown_path, grid_file)

    state.unknown_grid = grid_file
    state.unknown_abs = str(abs_v)
    state.unknown_norm = str(norm_v)

    if state.classification_method == "Геометричні центри (Евклідова відстань)":
        distances = []
        for center in state.cluster_centers:
            dist = np.sqrt(np.sum((np.array(norm_v) - np.array(center)) ** 2))
            distances.append(float(dist))

        min_dist = min(distances)
        predicted_class = distances.index(min_dist) + 1

        result_text = "МЕТОД: Геометричні центри\n\n"
        for i, d in enumerate(distances, start=1):
            result_text += f"Відстань до Класу {i}: {d:.4f}\n"

        result_text += f"\nВИСНОВОК: Образ належить до КЛАСУ {predicted_class} (Мінімальна відстань: {min_dist:.4f})"

    else:  # Метод ортогональних областей
        result_text = "МЕТОД: Ортогональні області (Min/Max)\n\n"
        matched_classes = []

        for class_idx, bounds in enumerate(state.cluster_min_max, start=1):
            fits = True
            out_of_bounds_count = 0
            for val, (min_b, max_b) in zip(norm_v, bounds):
                if not (min_b <= val <= max_b):
                    fits = False
                    out_of_bounds_count += 1

            if fits:
                matched_classes.append(class_idx)
                result_text += f"Клас {class_idx}: Порапляє в область (100% відповідність).\n"
            else:
                result_text += f"Клас {class_idx}: Не потрапляє (відхилень: {out_of_bounds_count}).\n"

        result_text += "\nВИСНОВОК: "
        if len(matched_classes) == 1:
            result_text += f"Образ однозначно належить до КЛАСУ {matched_classes[0]}."
        elif len(matched_classes) > 1:
            result_text += f"Образ потрапляє у перетин областей Класів: {matched_classes}."
        else:
            result_text += "Образ не потрапляє у жодну з областей."

    state.classification_result = result_text
    notify(state, "success", "Класифікацію виконано!")


# --- Інтерфейс Taipy ---
page = """
# Система розпізнавання образів з навчанням (Task 3)

---

## 1. Навчання системи (3 Класи)

<|layout|columns=1 1 1|
<|
### Клас 1
<|{c1_p1}|file_selector|label=Зразок 1.1|extensions=.bmp,.png,.jpg|>
<|{c1_g1}|image|width=120px|>

<|{c1_p2}|file_selector|label=Зразок 1.2|extensions=.bmp,.png,.jpg|>
<|{c1_g2}|image|width=120px|>

<|{c1_p3}|file_selector|label=Зразок 1.3|extensions=.bmp,.png,.jpg|>
<|{c1_g3}|image|width=120px|>

<|{c1_stats_text}|input|multiline=True|label=Статистика Класу 1|>
|>

<|
### Клас 2
<|{c2_p1}|file_selector|label=Зразок 2.1|extensions=.bmp,.png,.jpg|>
<|{c2_g1}|image|width=120px|>

<|{c2_p2}|file_selector|label=Зразок 2.2|extensions=.bmp,.png,.jpg|>
<|{c2_g2}|image|width=120px|>

<|{c2_p3}|file_selector|label=Зразок 2.3|extensions=.bmp,.png,.jpg|>
<|{c2_g3}|image|width=120px|>

<|{c2_stats_text}|input|multiline=True|label=Статистика Класу 2|>
|>

<|
### Клас 3
<|{c3_p1}|file_selector|label=Зразок 3.1|extensions=.bmp,.png,.jpg|>
<|{c3_g1}|image|width=120px|>

<|{c3_p2}|file_selector|label=Зразок 3.2|extensions=.bmp,.png,.jpg|>
<|{c3_g2}|image|width=120px|>

<|{c3_p3}|file_selector|label=Зразок 3.3|extensions=.bmp,.png,.jpg|>
<|{c3_g3}|image|width=120px|>

<|{c3_stats_text}|input|multiline=True|label=Статистика Класу 3|>
|>
|>

<br/>
<|Навчити систему|button|on_action=train_system|>
<br/><br/>
<|{training_status}|input|multiline=True|label=Статус навчання|>

---

## 2. Класифікація невідомого образу

<|layout|columns=1 1 1|
<|
### Невідомий образ
<|{unknown_path}|file_selector|label=Завантажити зображення|extensions=.bmp,.png,.jpg|>
<|{unknown_grid}|image|width=200px|>
<|{classification_method}|selector|lov=Геометричні центри (Евклідова відстань);Ортогональні області (Min/Max)|dropdown=True|label=Метод класифікації|>

<br/>
<|Розпізнати образ|button|on_action=classify_unknown|>
|>

<|
### Вектор невідомого образу
<|{unknown_abs}|input|multiline=True|label=Абсолютний вектор|>
<|{unknown_norm}|input|multiline=True|label=Нормований вектор|>
|>

<|
### Результат розпізнавання
<|{classification_result}|input|multiline=True|label=Деталі та висновок|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Навчання та Класифікація (Task 3)", dark_mode=False, port=5553)