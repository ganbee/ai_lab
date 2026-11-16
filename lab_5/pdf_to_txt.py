"""Завдання 1. Перетворює документи з data/ (PDF, DOCX) на чисті TXT у data/txt/.

Для кожного файлу:
  1) витягується текст (pdfplumber для PDF, python-docx для DOCX);
  2) прибираються службові артефакти (номери сторінок, службові символи);
  3) відсікаються частини, що не потрібні базі знань
     (анотації іншими мовами, списки літератури, дані про авторів).
"""
import re
from pathlib import Path

from docx import Document
import pdfplumber

DATA_DIR = Path("data")
TXT_DIR = DATA_DIR / "txt"
TXT_DIR.mkdir(parents=True, exist_ok=True)


def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def clean(text):
    """Спільне очищення тексту."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\uf000-\uf0ff]", "- ", text)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"(?<![.:;?!»\-])\n(?=[a-zа-яіїєґ0-9(«\"])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def trim_document_2(text):
    """Стаття: лишаємо заголовок і основний текст, без анотацій 3 мовами та списку літератури."""
    head_end = text.find("Анотації")
    body_start = text.find("Постановка проблеми")
    refs = text.find("Список використаних літературних джерел")
    head = text[:head_end] if head_end != -1 else ""
    body = text[body_start:refs] if body_start != -1 else text
    if refs == -1:
        body = text[body_start:]
    return head + "\n" + body


def trim_document_3(text):
    """DOCX: без блоку відомостей про авторів і без списку літератури."""
    title = "СУЧАСНИЙ СТАН ТА АКТУАЛЬНІ ПРОБЛЕМИ ЕРГОТЕРАПІЇ В УКРАЇНІ"
    start = text.find(title)
    end = text.find("\nЛітература")
    return text[start:end] if start != -1 and end != -1 else text


FILES = {
    "document_1.pdf": (read_pdf, None),
    "document_2.pdf": (read_pdf, trim_document_2),
    "document_3.docx": (read_docx, trim_document_3),
}


def main():
    for name, (reader, trimmer) in FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"Немає файлу: {path}")
            continue
        text = reader(path)
        if trimmer:
            text = trimmer(text)
        text = clean(text)
        out = TXT_DIR / f"{path.stem}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"{name} -> {out}  ({len(text)} символів)")


if __name__ == "__main__":
    main()