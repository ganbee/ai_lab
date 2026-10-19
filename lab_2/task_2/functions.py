from PIL import Image, ImageDraw
import numpy as np

GRID_ROWS = 5
GRID_COLS = 5
N_FEATURES = 25


def process_image_for_lab2(image_path: str, temp_file_path: str):
    """
    Універсальна обробка BMP для будь-яких типів шрифтів та товщин ліній.
    """
    if not image_path:
        return None, None, None, None

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

        # сітка
        grid_img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(grid_img)

        for col in range(1, GRID_COLS):
            x = int(col * cell_w)
            draw.line((x, 0, x, height), fill="red", width=2)

        for row in range(1, GRID_ROWS):
            y = int(row * cell_h)
            draw.line((0, y, width, y), fill="red", width=2)

        grid_img.save(temp_file_path)

        # 1. нормований вектор (Хеммінг)
        total_black = sum(abs_vec)
        norm_vec = [round(v / total_black, 4) for v in abs_vec] if total_black > 0 else [0.0] * N_FEATURES

        # 2. відносна бінаризація для Хопфілда (ураховує геометрію форми, а не товщину)
        max_val = max(abs_vec) if max(abs_vec) > 0 else 1
        bin_vec = [1 if (v / max_val) >= 0.25 else -1 for v in abs_vec]

        return abs_vec, norm_vec, bin_vec, temp_file_path
    except Exception as e:
        print(f"Помилка обробки зображення: {e}")
        return None, None, None, None


# Мережа Хопфілда
def train_hopfield(patterns: list[list[int]]) -> np.ndarray:
    p = np.array(patterns, dtype=float)
    m, n = p.shape
    w = np.zeros((n, n))

    for k in range(m):
        w += np.outer(p[k], p[k])

    np.fill_diagonal(w, 0)
    return w / float(n)


def classify_hopfield(x_test_bin: list[int], w_matrix: np.ndarray, max_iter: int = 100) -> tuple[list[int], int]:
    y = np.array(x_test_bin, dtype=float)
    n = len(y)
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        y_old = np.copy(y)
        
        for i in range(n):
            s = np.dot(w_matrix[i], y)
            if s > 0:
                y[i] = 1
            elif s < 0:
                y[i] = -1

        if np.array_equal(y, y_old):
            break

    return y.astype(int).tolist(), iterations


# Мережа Хеммінга
def train_hamming(patterns_norm: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    p = np.array(patterns_norm, dtype=float)
    m, n = p.shape
    w_1 = p / 2.0
    b_1 = np.full((m,), n / 2.0)
    return w_1, b_1


def classify_hamming(x_test_norm: list[float], w_1: np.ndarray, b_1: np.ndarray, max_iter: int = 200) -> tuple[int, list[float], int]:
    m = w_1.shape[0]
    x = np.array(x_test_norm, dtype=float)

    y1 = np.dot(w_1, x) + b_1

    e = 1.0 / float(m + 1)
    y2 = np.copy(y1)
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        y2_prev = np.copy(y2)
        for j in range(m):
            sum_other = np.sum(y2_prev) - y2_prev[j]
            y2[j] = max(0.0, y2_prev[j] - e * sum_other)

        active = np.where(y2 > 0)[0]
        if len(active) <= 1 or np.allclose(y2, y2_prev, atol=1e-3):
            break

    winner_class = int(np.argmax(y2))
    return winner_class, [round(float(v), 2) for v in y2.tolist()], iterations