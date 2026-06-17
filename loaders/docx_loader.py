from docx import Document

def ingest_docx(filepath):

    print(f"Ingesting: {filepath}")

    doc = Document(filepath)

    text = []

    for para in doc.paragraphs:

        if para.text.strip():

            text.append(
                para.text.strip()
            )

    return "\n".join(text)


def chunk_docx(
    text,
    source_file
):

    return [
        {
            "chunk_id": "chunk_0",
            "text": text,
            "metadata": {
                "source": source_file,
                "section": "Document",
                "parent_section": "Document",
                "chunk_index": 0
            }
        }
    ]