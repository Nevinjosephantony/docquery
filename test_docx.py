from loaders.docx_loader import ingest_docx

text = ingest_docx(
    r"C:\Users\user\Desktop\mistakes.docx"
)

print(text)