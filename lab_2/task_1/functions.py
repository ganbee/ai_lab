from PIL import Image, ImageDraw
import numpy as np

GRID_ROWS = 5
GRID_COLS = 5
N_FEATURES = 25


def process_image_and_save_grid(image_path: str, temp_file_path: str) -> tuple[list[int] | None, list[float] | None, str | None]:
    """
    Зчитує BMP, малює червону сітку 5х5, зберігає її у temp_file_path
    та повертає АБСОЛЮТНИЙ і НОРМОВАНИЙ вектори ознак.
    """
    if not image_path:
        return None, None, None

    try:
        image = Image.open(image_path)
        image_gray = image.convert("L")
        width, height = image_gray.size
        pixels = np.array(image_gray)

        cell_w = width / GRID_COLS
        cell_h = height / GRID_ROWS
        abs_vec = []

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                cell = pixels[y1:y2, x1:x2]
                abs_vec.append(int(np.sum(cell < 128)))

        # червона сітка 5х5
        grid_img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(grid_img)

        for col in range(1, GRID_COLS):
            x = int(col * cell_w)
            draw.line((x, 0, x, height), fill="red", width=2)

        for row in range(1, GRID_ROWS):
            y = int(row * cell_h)
            draw.line((0, y, width, y), fill="red", width=2)

        grid_img.save(temp_file_path)

        # нормування вектора
        total_black = sum(abs_vec)
        norm_vec = [round(v / total_black, 4) for v in abs_vec] if total_black > 0 else [0.0] * N_FEATURES

        return abs_vec, norm_vec, temp_file_path
    except Exception as e:
        print(f"Помилка обробки зображення: {e}")
        return None, None, None


def calculate_perceptron_output(x: list[float], weights: list[float], bias: float) -> tuple[float, int]:
    """
    S = sum(w_i * x_i)
    y = +1 якщо S > bias, інакше -1
    """
    w = np.array(weights, dtype=float)
    x_vec = np.array(x, dtype=float)
    s = float(np.dot(w, x_vec))
    y = 1 if s > bias else -1
    return round(s, 4), y


def update_weights(
    weights: list[float], 
    x: list[float], 
    desired_class: int, 
    predicted_class: int, 
    learning_rate: float
) -> list[float]:
    """
    W_new = W + learning_rate * (d - y) * X
    """
    w = np.array(weights, dtype=float)
    x_vec = np.array(x, dtype=float)
    error = desired_class - predicted_class
    w_new = w + learning_rate * error * x_vec
    return [round(float(val), 4) for val in w_new.tolist()]