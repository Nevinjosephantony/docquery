from pathlib import Path

def route_file(filepath: str):
    ext = Path(filepath).suffix.lower()

    if ext in [".pdf", ".docx"]:
        return "rag_pipeline"

    elif ext in [".csv", ".xlsx"]:
        return "sql_pipeline"

    else:
        raise ValueError(f"Unsupported file type: {ext}")