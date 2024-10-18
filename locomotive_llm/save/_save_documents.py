from docx import Document
from pathlib import Path


def write_docx(translated_text, output_path):
    doc = Document()
    for paragraph in translated_text.split("\n"):
        doc.add_paragraph(paragraph)
    doc.save(output_path)


def write_docx_with_formatting(translated_elements, output_path):
    doc = Document()
    doc.map_translations(translated_elements)
    doc.save(output_path)


def write_txt(translated_text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(translated_text)


def write_doc(translated_text, output_file, output_format=None, preserve_formatting=False):
    if output_format is None:
        output_format = Path(output_file).suffix
    if output_format == ".docx":
        if preserve_formatting:
            write_docx_with_formatting(translated_text, output_file)
        else:
            write_docx(translated_text, output_file)
    elif output_format == ".txt":
        write_txt(translated_text, output_file)
