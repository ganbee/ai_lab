"""Використання: python pdf_to_txt.py шлях/до/статті.pdf
Витягує текст із PDF і зберігає в data/text2.txt"""
import sys
from pathlib import Path
from pypdf import PdfReader

src = Path(sys.argv[1])
reader = PdfReader(src)
text = "\n".join((page.extract_text() or "") for page in reader.pages)
Path("data").mkdir(exist_ok=True)
Path("data/text2.txt").write_text(text, encoding="utf-8")
print(f"Сторінок: {len(reader.pages)}, символів: {len(text)}")