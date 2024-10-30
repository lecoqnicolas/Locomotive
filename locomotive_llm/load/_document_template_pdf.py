import pdfplumber
from fpdf import FPDF


class PDFDocumentTemplate:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._elements = []
        self._load_pdf()

    def _load_pdf(self):
        with pdfplumber.open(self._file_path) as pdf:
            for page in pdf.pages:
                # Extracting paragraphs
                # Extracting text by characters with font and size
                line_elements = []
                previous_char_y = None
                '''if text:
                    for line in text.split("\n"):
                        if line.strip(): 
                            self._elements.append({'type': 'paragraph', 'content': line.strip(), 'font': 'Arial', 'size': 12})'''
                            #format same than input
                        
                for char in page.chars:
                    font = char.get('fontname', "Arial")
                    size = char.get('size', 12)
                    current_char_y = char['top'] 
                    if previous_char_y is not None and abs(current_char_y - previous_char_y) > 5:
                        line_content = ''.join([el['char'] for el in line_elements])
                        if line_content.strip():
                            self._elements.append({
                                'type': 'paragraph',
                                'content': line_content,
                                'font': line_elements[0]['font'],
                                'size': line_elements[0]['size']
                            })
                        line_elements = []
                    # Add character to the current line
                    line_elements.append({'char': char['text'], 'font': font, 'size': size})
                    previous_char_y = current_char_y  # Update Y position for the next iteration

                # Add the last line after loop ends
                if line_elements:
                    line_content = ''.join([el['char'] for el in line_elements])
                    if line_content.strip():
                        self._elements.append({
                            'type': 'paragraph',
                            'content': line_content,
                            'font': line_elements[0]['font'],
                            'size': line_elements[0]['size']
                        })

                # Extracting tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            for cell in row:
                                self._elements.append({'type': 'table', 'content': cell})

    def get_content(self):
        return self._elements

    def map_translations(self, translations):
        element_count = 0
        for element in self._elements:
            if element['type'] == 'paragraph' and element_count < len(translations):
                if translations[element_count].strip():
                    element['content'] = translations[element_count].strip()
                element_count += 1
    def sanitize_text(self, text):
        replacements = {
            '\u2018': "'",
            '\u2019': "'",
            '\u201c': '"',
            '\u201d': '"',
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text

    def save(self, output_file):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        for element in self._elements:
            content = element['content']
            try:
                font_name = element.get('font', "Arial").split(",")[0]
                font_style = "B" if "bold" in element.get('font', "").lower() else ""
                pdf.set_font(font_name, style=font_style, size=element.get('size', 12))
            except RuntimeError:
                pdf.set_font("Arial", size=element.get('size', 12))

            if element['type'] == 'paragraph':
                sanitized_content = self.sanitize_text(content)
                pdf.multi_cell(0, 10, sanitized_content, align='L')
                pdf.ln()
            elif element['type'] == 'table':
                table_content = content.splitlines()
                for row in table_content:
                    sanitized_row = self.sanitize_text(row)
                    pdf.multi_cell(0, 10, sanitized_row, align='L')

        pdf.output(output_file)
