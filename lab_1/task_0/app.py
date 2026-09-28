from taipy.gui import Gui, notify

input_image = None
output_image = None
message = ""
show_result = False
reset_key = 0


def show_image(state):
    # Перевіряємо, чи був обраний файл
    if not state.input_image:
        state.message = "Будь ласка, завантажте зображення."
        state.show_result = False
        notify(state, "warning", state.message)
    else:
        state.message = "Зображення успішно завантажено!"
        # Присвоюємо шлях файлу для відображення
        state.output_image = state.input_image
        state.show_result = True
        notify(state, "success", state.message)


page = """
# Task 0 - Image Uploader

<|{input_image}|file_selector|label=Завантажте зображення|extensions=.png,.jpg,.jpeg|>

<|Показати зображення|button|on_action=show_image|>

### Повідомлення
<|{message}|text|>

<|{output_image}|image|render={show_result}|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Task 0 - Image Uploader", port=5550, debug=True)