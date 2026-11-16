"""Збирає чернетку звіту до ЛР №5 (Word) із файлів results/*.json.

Запуск: python build_report.py  ->  results/report_lab5.docx

Скрипт підставляє всі таблиці й числа з результатів експериментів. Місця, які треба
дописати вручну (аналіз, висновки), позначені жовтим кольором: [ДОПИСАТИ: ...].
Якщо якогось файлу results/taskN.json ще немає, відповідний розділ отримає позначку
про це, а решта звіту збереться.
"""
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from task1 import CHUNK_SIZE, OVERLAP

RESULTS = Path("results")
TXT_DIR = Path("data") / "txt"
OUT = RESULTS / "report_lab5.docx"
FONT = "Times New Roman"

DOC_INFO = {
    "document_1.pdf": "Розділ 1 навчально-методичного посібника «Роль ерготерапії у процесі "
                      "реабілітації» (с. 6–35). [ДОПИСАТИ: автори, видавництво, рік]",
    "document_2.pdf": "Гавалко І., Бас О. Ерготерапія в реабілітації хворих з перенесеним "
                      "інсультом. Львівський державний університет фізичної культури (стаття).",
    "document_3.docx": "Клеценко Л. В., Вишар Є. В. Сучасний стан та актуальні проблеми "
                       "ерготерапії в Україні (стаття).",
}

CONTROL_QUESTIONS = [
    "Що таке RAG?", "Для чого використовується RAG?", "Що таке embedding?",
    "Для чого текст перетворюється на embedding?", "Що таке vector store?",
    "Що таке retrieval?", "Що таке chunking?", "Навіщо документи розділяють на chunks?",
    "Яку роль відіграє LLM у RAG-системі?",
    "Чим RAG відрізняється від звичайного запиту до LLM?", "Які переваги RAG?",
    "Які обмеження має RAG?", "Чому RAG може повертати неправильну відповідь?",
    "Що відбувається, якщо необхідної інформації немає у векторному сховищі?",
    "Чим RAG відрізняється від Fine-tuning?",
]


def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f"УВАГА: немає {path}, відповідний розділ буде порожнім")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def set_font(run, size=None, bold=None, italic=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def para(doc, text="", size=12, bold=False, italic=False, align=None, space_after=6):
    p = doc.add_paragraph()
    set_font(p.add_run(text), size=size, bold=bold, italic=italic)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if align:
        p.alignment = align
    return p


def heading(doc, text):
    p = para(doc, text, size=14, bold=True, space_after=6)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.keep_with_next = True
    return p


def placeholder(doc, hint):
    """Жовте виділення: місце, яке студентка дописує сама."""
    p = doc.add_paragraph()
    run = p.add_run(f"[ДОПИСАТИ: {hint}]")
    set_font(run, size=12)
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    p.paragraph_format.space_after = Pt(6)
    return p


def missing(doc, filename, script):
    para(doc, f"[Немає results/{filename}. Запустіть {script} і перезберіть звіт.]",
         italic=True)


def clean_md(text):
    """Прибирає markdown-розмітку відповідей моделі (жирний шрифт, маркери)."""
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", text)
    return text.strip()


def shade(cell, color="D9D9D9"):
    tc_pr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def table(doc, headers, rows, widths_cm, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        set_font(cell.paragraphs[0].add_run(h), size=font_size, bold=True)
        shade(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            lines = str(value).split("\n")
            for k, line in enumerate(lines):
                p = cells[i].paragraphs[0] if k == 0 else cells[i].add_paragraph()
                set_font(p.add_run(line), size=font_size)
                p.paragraph_format.space_after = Pt(2)
    for i, w in enumerate(widths_cm):
        t.columns[i].width = Cm(w)          
        for row in t.rows:
            row.cells[i].width = Cm(w)     
    tr_pr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    tr_pr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def short(text, n):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def section_title(doc):
    para(doc, "Лабораторна робота № 5", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Тема: Побудова простої RAG-системи", size=14, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Предметна область: фізична реабілітація та ерготерапія", size=12,
         italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

    heading(doc, "1. Мета роботи")
    para(doc, "Ознайомитися з принципом роботи Retrieval-Augmented Generation (RAG); навчитися "
              "використовувати зовнішню базу документів для пошуку інформації та формування "
              "відповідей великою мовною моделлю; набути практичних навичок роботи з "
              "embeddings, векторним сховищем та пошуком релевантних фрагментів тексту.")

    heading(doc, "2. Короткі теоретичні відомості про RAG")
    placeholder(doc, "2–3 абзаци: що таке RAG, embedding, vector store, retrieval "
                     "(можна взяти з методички та відповідей на контрольні запитання)")


def section_documents(doc, chunks):
    heading(doc, "3. Опис набору документів")
    if not chunks:
        return missing(doc, "task1.json", "task1.py")
    para(doc, "Для бази знань використано три відкриті документи українською мовою за темою "
              "ерготерапії (інсульт, повсякденна діяльність, стан галузі в Україні):")
    rows = []
    for source in sorted({c["source"] for c in chunks}):
        items = [c for c in chunks if c["source"] == source]
        txt = TXT_DIR / (Path(source).stem + ".txt")
        n_chars = len(txt.read_text(encoding="utf-8")) if txt.exists() \
            else sum(c["n_chars"] for c in items)
        rows.append([source, DOC_INFO.get(source, ""), Path(source).suffix[1:].upper(),
                     n_chars, len(items)])
    rows.append(["Разом", "", "", sum(r[3] for r in rows), len(chunks)])
    table(doc, ["Файл", "Опис", "Формат", "Символів", "Chunks"], rows, [2.8, 8.2, 1.7, 2.0, 1.7])


def section_tools(doc, task2, task4_prompts):
    heading(doc, "4. Опис використаних інструментів")
    llm = (task4_prompts[0]["model"] if task4_prompts else "[ДОПИСАТИ: модель Gemini]")
    emb = task2["embedding_model"] if task2 else "gemini-embedding-001"
    dim = task2["dimension"] if task2 else 768
    rows = [
        ["Мова програмування", "Python 3.12"],
        ["Робота з документами", "pdfplumber (PDF), python-docx (DOCX)"],
        ["Chunking", f"за реченнями, до {CHUNK_SIZE} символів, перекриття ~{OVERLAP} символів"],
        ["Embedding-модель", f"{emb} (Gemini API), розмірність {dim}; "
                              "task_type: RETRIEVAL_DOCUMENT для chunks, RETRIEVAL_QUERY для запитів"],
        ["Векторне сховище", "ChromaDB (PersistentClient), косинусна відстань"],
        ["Мовна модель (LLM)", f"{llm} (Gemini API), temperature = 0"],
        ["Кількість знайдених фрагментів", "top-k = 3"],
    ]
    table(doc, ["Компонент", "Опис"], rows, [4.5, 12.0], font_size=10)


def section_scheme(doc):
    heading(doc, "5. Схема роботи розробленої системи")
    scheme = [
        "Документи (PDF, DOCX)  -> pdf_to_txt.py  ->  чистий текст (TXT)",
        "      ↓  task1.py: розбиття на chunks + збереження джерела",
        "Chunks (текст + source)",
        "      ↓  task2.py: Gemini Embeddings (RETRIEVAL_DOCUMENT)",
        "Векторне сховище ChromaDB (текст + вектор + джерело)",
        "",
        "Запит користувача",
        "      ↓  Gemini Embeddings (RETRIEVAL_QUERY)",
        "Пошук top-3 за косинусною відстанню (task3.py)",
        "      ↓",
        "Контекст + prompt (task4.py)",
        "      ↓",
        "Gemini (генерація, temperature = 0)",
        "      ↓",
        "Відповідь + джерела",
    ]
    for line in scheme:
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Courier New"
        r.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph()
    placeholder(doc, "вставити програмний код або посилання на репозиторій "
                     "(main.py, pdf_to_txt.py, task1.py … task6.py)")


def section_chunks_embeddings(doc, chunks, task2):
    heading(doc, "6. Chunking і embeddings (завдання 2–4)")
    if not chunks:
        return missing(doc, "task1.json", "task1.py")
    sizes = [c["n_chars"] for c in chunks]
    para(doc, f"Усього отримано {len(chunks)} chunks. Довжина chunk: мінімальна {min(sizes)}, "
              f"середня {sum(sizes) // len(sizes)}, максимальна {max(sizes)} символів. "
              "Для кожного chunk збережено джерело (структура «документ → chunks»):")
    rows = []
    for source in sorted({c["source"] for c in chunks}):
        items = [c for c in chunks if c["source"] == source]
        rows.append([source, f"{items[0]['chunk_id']} … {items[-1]['chunk_id']}", len(items)])
    table(doc, ["Документ", "Chunks (id)", "Кількість"], rows, [5.0, 8.0, 3.5], font_size=10)

    para(doc, "Приклад відповідності chunk → джерело:")
    ex = [[c["chunk_id"], c["source"], short(c["text"], 90)] for c in chunks[:3]]
    table(doc, ["Chunk", "Джерело", "Початок тексту"], ex, [2.5, 3.0, 11.0])

    if task2:
        e = task2["example"]
        para(doc, f"Кожен chunk перетворено на embedding розмірністю {task2['dimension']}. "
                  f"Приклад ({e['chunk_id']}, {e['source']}):")
        para(doc, "Текст: " + short(e["text"], 200), italic=True)
        vec = ", ".join(f"{x:.3f}" for x in e["embedding_first_10"])
        para(doc, f"Embedding: [{vec}, …]")
        para(doc, f"У векторному сховищі ChromaDB збережено {task2['n_records']} записів "
                  "(текст, embedding, назва документа).")
    else:
        missing(doc, "task2.json", "task2.py")
    placeholder(doc, "1–2 речення: embedding використовується для порівняння змісту, а не для "
                     "генерації тексту")


def section_retrieval(doc, task3):
    heading(doc, "7. Пошук релевантних фрагментів (завдання 5)")
    if not task3:
        return missing(doc, "task3.json", "task3.py")
    rows = []
    for i, item in enumerate(task3, start=1):
        top = item["hits"][0]
        rows.append([i, item["query"], f"{top['chunk_id']}\n{short(top['text'], 110)}",
                     top["source"], top["distance"],
                     ", ".join(item["expected_sources"]),
                     "так" if item["top1_correct"] else "ні"])
    table(doc, ["№", "Запит", "Найближчий chunk", "Джерело", "Відстань", "Очікуване джерело",
                "Збіг"], rows, [0.8, 3.2, 4.6, 2.6, 1.4, 2.6, 1.2], font_size=8)
    n = len(task3)
    t1 = sum(i["top1_correct"] for i in task3)
    t3 = sum(i["top3_correct"] for i in task3)
    para(doc, f"Джерело першого результату збіглося з очікуваним у {t1} із {n} запитів; "
              f"очікуване джерело є серед трьох найближчих chunks у {t3} із {n}.")
    placeholder(doc, "аналіз retrieval: розбиття відповіді між сусідніми chunks "
                     "(SMART — chunk_010 і chunk_011; стратегії — chunk_009 і chunk_010), "
                     "вузький діапазон відстаней, слабкі треті результати")


def answer_table(doc, rows_src):
    rows = []
    for r in rows_src:
        frag = "\n".join(f"{c['chunk_id']} ({c['distance']}): {short(c['preview'], 140)}"
                         for c in r["chunks"])
        rows.append([r["n"], r["query"], frag, "\n".join(r["sources"]),
                     clean_md(r["answer"])])
    table(doc, ["№", "Запит", "Знайдений фрагмент", "Джерело", "Відповідь LLM"], rows,
          [0.8, 2.5, 4.6, 2.7, 6.4], font_size=8)


def section_rag(doc, task4):
    heading(doc, "8. Відповіді LLM на основі знайденого контексту (завдання 6–7)")
    para(doc, "Знайдені фрагменти передавалися Gemini в prompt за шаблоном з методички "
              "(«Використовуючи наведений контекст, дай відповідь на запитання… Якщо відповіді "
              "немає в контексті, повідом про це»).")
    if not task4:
        return missing(doc, "task4.json", "task4.py")
    answer_table(doc, task4)
    placeholder(doc, "аналіз відповідей: усі твердження підтверджені контекстом; неповна "
                     "відповідь у запиті 7 (названо лише АТО з кількох причин); у запиті 6 "
                     "модель дописала id chunk у дужках; зайве markdown-форматування")


def section_limits(doc, task5):
    heading(doc, "9. Запитання, відповіді на які немає в базі (завдання 8)")
    if not task5:
        return missing(doc, "task5.json", "task5.py")
    low, high = task5["in_base_top1_distance_range"]
    para(doc, f"Для запитань із відповіддю в базі відстань до найближчого chunk становила "
              f"від {low} до {high}.")
    rows = []
    for q in task5["questions"]:
        rows.append([q["n"], q["type"], q["query"],
                     f"{q['chunks'][0]['chunk_id']}, {q['best_distance']}"
                     + (" (вище за максимум з бази)" if q["above_in_base_max"] else ""),
                     clean_md(q["answer"]),
                     "так" if q["says_no_info_heuristic"] else "ні"])
    table(doc, ["№", "Тип", "Запит", "Найближчий chunk, відстань", "Відповідь LLM",
                "Визнала, що інформації немає (евристика)"], rows,
          [0.8, 2.3, 3.0, 3.0, 5.6, 1.8], font_size=8)
    placeholder(doc, "аналіз: чи вигадує модель; чи визнає відсутність інформації; чи можна "
                     "було відсіяти запит порогом відстані; висновок про важливість перевірки "
                     "знайденого контексту (оцінюйте вручну, евристика наближена)")


def section_compare(doc, task6):
    heading(doc, "10. Порівняння LLM без RAG та з RAG (додаткове завдання)")
    if not task6:
        return missing(doc, "task6.json", "task6.py")
    para(doc, "Однакові запитання ставилися Gemini напряму (без контексту) і через RAG; "
              "параметри моделі однакові (temperature = 0).")
    rows = []
    for r in task6:
        rows.append([r["n"], r["query"], r["reference"],
                     clean_md(r["answer_without_rag"]), clean_md(r["answer_with_rag"]),
                     ("так" if r["key_facts_in_answer_without_rag"] else "ні") + " / "
                     + ("так" if r["key_facts_in_answer_with_rag"] else "ні")])
    table(doc, ["№", "Запитання", "Еталон (документи)", "Без RAG", "З RAG",
                "Ключові факти: без / з RAG"], rows, [0.7, 2.6, 2.8, 4.4, 4.4, 1.9], font_size=8)
    n = len(task6)
    w = sum(r["key_facts_in_answer_without_rag"] for r in task6)
    r_ = sum(r["key_facts_in_answer_with_rag"] for r in task6)
    para(doc, f"Автоперевірка наявності ключових фактів: без RAG {w} із {n}, з RAG {r_} із {n}.")
    placeholder(doc, "порівняння за критеріями методички: точність, повнота, відповідність "
                     "наданим документам, наявність вигаданої інформації")


def section_end(doc):
    heading(doc, "11. Висновки")
    placeholder(doc, "підсумок: що реалізовано, якість retrieval, роль контексту, обмеження RAG, "
                     "порівняння з RAG і без RAG; зауваження про перевірку відповідей LLM "
                     "у медичній темі")
    heading(doc, "Контрольні запитання")
    for i, q in enumerate(CONTROL_QUESTIONS, start=1):
        para(doc, f"{i}. {q}", bold=True, space_after=2)
        placeholder(doc, "відповідь")


def main():
    chunks = load("task1.json")
    task2 = load("task2.json")
    task3 = load("task3.json")
    task4 = load("task4.json")
    task5 = load("task5.json")
    task6 = load("task6.json")
    prompts = load("task4_prompts.json")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(2)
    doc.styles["Normal"].font.name = FONT
    doc.styles["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    section_title(doc)
    section_documents(doc, chunks)
    section_tools(doc, task2, prompts)
    section_scheme(doc)
    section_chunks_embeddings(doc, chunks, task2)
    section_retrieval(doc, task3)
    section_rag(doc, task4)
    section_limits(doc, task5)
    section_compare(doc, task6)
    section_end(doc)

    doc.save(OUT)
    print(f"Збережено: {OUT}")


if __name__ == "__main__":
    main()