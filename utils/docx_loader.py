from docx import Document


def extract_text_from_docx(docx_path):
    doc = Document(docx_path)

    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text