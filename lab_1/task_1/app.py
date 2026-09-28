from taipy.gui import Gui, notify
from PIL import Image, ImageDraw
import numpy as np
import os

# --- Константи ---
GRID_ROWS = 5
GRID_COLS = 5
TEMP_IMAGE_PATH = "temp_grid_output.png"

# --- Змінні стану ---
image_path = None
grid_image = None
absolute_text = "[]"
normalized_text = "[]"

# --- Основна функція обробки ---
def process_image(state):
    if not state.image_path:
        notify(state, "error", "Будь ласка, завантажте зображення у форматі BMP.")
        return

    try:
        # 1. Завантаження та перетворення у відтінки сірого
        image = Image.open(state.image_path)
        image_gray = image.convert("L")
        width, height = image_gray.size
        pixels = np.array(image_gray)
        
        cell_width = width / GRID_COLS
        cell_height = height / GRID_ROWS
        
        absolute_vector = []
        
        # 2. Обчислення абсолютного вектора ознак
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

        # 3. Обчислення нормованого вектора
        if total_black_pixels > 0:
            normalized_vector = [
                round(value / total_black_pixels, 4)
                for value in absolute_vector
            ]
        else:
            normalized_vector = [0.0] * len(absolute_vector)

        # 4. Малювання сітки
        grid_img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(grid_img)

        for col in range(1, GRID_COLS):
            x = int(col * cell_width)
            draw.line((x, 0, x, height), fill="red", width=2)

        for row in range(1, GRID_ROWS):
            y = int(row * cell_height)
            draw.line((0, y, width, y), fill="red", width=2)

        # 5. Збереження зображення у файл для чистого відображення в Taipy
        grid_img.save(TEMP_IMAGE_PATH)
        
        # Оновлення стану
        state.grid_image = TEMP_IMAGE_PATH
        state.absolute_text = str(absolute_vector)
        state.normalized_text = str(normalized_vector)
        
        notify(state, "success", "Вектори ознак успішно побудовано!")

    except Exception as e:
        notify(state, "error", f"Виникла помилка: {e}")

# --- Макет сторінки ---
page = """
# Система побудови вектора ознак (Taipy)

Завантажте чорно-біле зображення у форматі **BMP**.

Програма розділить його на сітку **5×5**, порахує кількість чорних пікселів у кожній області та сформує абсолютний і нормований вектори ознак.

<|layout|columns=1 1|
<|
### 1. Введення даних
<|{image_path}|file_selector|label=Завантажити BMP-зображення|extensions=.bmp|>

<br/>
<|Побудувати вектор ознак|button|on_action=process_image|>
|>

<|
### 2. Зображення із сіткою 5×5
<|{grid_image}|image|width=300px|>
|>
|>

---

## Результати

<|layout|columns=1 1|
<|
<|{absolute_text}|input|multiline=True|class_name=fullwidth|label=Абсолютний вектор ознак|>
|>

<|
<|{normalized_text}|input|multiline=True|class_name=fullwidth|label=Нормований вектор ознак|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Вектор ознак зображення (Task 1)", dark_mode=False, port=5551)