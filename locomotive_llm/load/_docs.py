import logging
from pathlib import Path

from docx import Document
from langchain_community.document_loaders import TextLoader

from ._docxloader import DocxLoader


def read_docx_with_langchain(file_path):
    """Reads a .docx file using the custom LangChain loader."""
    loader = DocxLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


def read_docx_with_formatting(file_path):
    """Reads a .docx file and preserves the structure (tables, paragraphs, etc.)."""
    doc = Document(file_path)
    elements = []

    for para in doc.paragraphs:
        elements.append({'type': 'paragraph', 'content': para.text})

    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        elements.append({'type': 'table', 'content': table_data})

    return elements


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_txt_langchain(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


def read_doc(file_path, use_langchain_txt=False, preserve_formatting=True):
    """Reads the ground truth document (either .docx or .txt) using appropriate loaders."""
    file_extension = Path(file_path).suffix
    if file_extension == ".docx":
        if preserve_formatting:
            return read_docx_with_formatting(file_path)
        return read_docx_with_langchain(file_path)  # Use the custom loader for .docx
    elif file_extension == ".txt":
        if use_langchain_txt:
            return read_txt_langchain(file_path)
        else:
            return read_txt(file_path)  # Use the standard function for .txt
    else:
        raise NotImplementedError(f"Unsupported file format {file_extension} for file {file_path}, "
                                  f"Only .docx and .txt files are supported.")
