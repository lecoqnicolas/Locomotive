from docx import Document


class DocumentTemplate:
    def __init__(self, file_path: str):
        self._doc = Document(file_path)
        self._nb_para = None
        self._nb_table = None

    def get_content(self):
        elements = []
        for para in self._doc.paragraphs:
            elements.append({'type': 'paragraph', 'content': para.text})
        self._nb_para = len(elements)

        for table in self._doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    elements.append({'type': 'table', 'content': cell.text})
        return elements

    def map_translations(self, translations):
        for i, para in enumerate(self._doc.paragraphs):
            if len(translations[i]) > 0:
                para.text = translations[i]

        for j, table in enumerate(self._doc.tables):
            for k, row in enumerate(table.rows):
                for l, cell in enumerate(row.cells):
                    if len(translations[self._nb_para+j]) > 0:
                        cell.text = translations[self._nb_para+j+k+l]

    def save(self, file_path):
        self._doc.save(file_path)
