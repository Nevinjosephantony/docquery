import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.document import PictureItem
from llm import call_llm

def ingest_pdf(file_path):
    print(f"Ingesting: {file_path}")
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )
    
    result = converter.convert(file_path)
    return result.document

def enrich_document(doc):
    markdown = doc.export_to_markdown()
    
    summary = call_llm(f"""Read this document and return JSON with:
- document_type
- summary
- main_topics

Document:
{markdown[:3000]}

Return only valid JSON.""")
    
    return {
        "markdown": markdown,
        "enrichment": summary
    }

def extract_images(doc, output_folder="output/images"):
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = []
    count = 0
    
    for i, (item, level) in enumerate(doc.iterate_items()):
        if isinstance(item, PictureItem):
            try:
                image = item.get_image(doc)
                if image:
                    count += 1
                    image_path = os.path.join(output_folder, f"image_{count}.png")
                    image.save(image_path)
                    image_paths.append(image_path)
            except Exception as e:
                print(f"Failed {i+1}: {e}")
    
    print(f"Total saved: {len(image_paths)}")
    return image_paths

def _make_chunk(index, text, h1, h2, source):
    section = h2 if h2 else h1
    clean_text = text.replace("<!-- image -->", "").strip()
    return {
        "chunk_id": f"chunk_{index}",
        "text": clean_text,
        "metadata": {
            "source": source,
            "section": section,
            "parent_section": h1,
            "chunk_index": index
        }
    }

def chunk_document(doc, source_file):
    markdown = doc.export_to_markdown()
    chunks = []
    
    current_h1 = "Document"
    current_h2 = None
    current_text = ""
    chunk_index = 0
    
    for line in markdown.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            if current_text.strip():
                chunks.append(_make_chunk(
                    chunk_index, current_text,
                    current_h1, current_h2, source_file
                ))
                chunk_index += 1
                current_text = ""
            current_h1 = line[2:].strip()
            current_h2 = None

        elif line.startswith("## "):
            if current_text.strip():
                chunks.append(_make_chunk(
                    chunk_index, current_text,
                    current_h1, current_h2, source_file
                ))
                chunk_index += 1
                current_text = ""
            current_h2 = line[3:].strip()

        else:
            current_text += line + "\n"
    
    if current_text.strip():
        chunks.append(_make_chunk(
            chunk_index, current_text,
            current_h1, current_h2, source_file
        ))
    
    # Remove chunks with no real text after image tag removal
    chunks = [c for c in chunks if c["text"]]
    
    return chunks

if __name__ == "__main__":
    doc = ingest_pdf(r"D:\chatbot\aonz\claude free Updated doc.pdf")
    enriched = enrich_document(doc)
    images = extract_images(doc)
    chunks = chunk_document(doc, "claude_free.pdf")
    
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\n[{chunk['metadata']['section']}]: {chunk['text'][:100]}")
    print("\nIMAGES:", images)
    print("\nLLM ENRICHMENT:", enriched["enrichment"])