# DocQuery — Document Question Answering with RAG

DocQuery is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content.

The application extracts text from documents, creates embeddings, stores them in ChromaDB, retrieves relevant sections for a user query, and uses an LLM to generate an answer grounded in the retrieved context.

## Features

* PDF and DOCX document ingestion
* Text extraction and document processing
* Semantic embeddings using `mixedbread-ai/mxbai-embed-large-v1`
* Vector storage and similarity search with ChromaDB
* Top-k relevant chunk retrieval
* LLM-based question answering using Groq
* Source-aware answers based on retrieved document context
* Rejects questions when the required information is not present in the retrieved context
* Simple application interface for document querying

## Architecture

```text
                    ┌─────────────────┐
                    │  Upload Document│
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Document Ingestion  │
                  │ PDF / DOCX          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Text Extraction &   │
                  │ Chunking            │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Embedding Model     │
                  │ mxbai-embed-large   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     ChromaDB        │
                  │   Vector Storage    │
                  └──────────┬──────────┘
                             │
                      User Question
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Semantic Retrieval  │
                  │     Top-K Chunks    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      Groq LLM       │
                  │   GPT-OSS 20B       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Grounded Answer +   │
                  │ Sources             │
                  └─────────────────────┘
```

## Tech Stack

* **Python**
* **ChromaDB** — vector database
* **Hugging Face** — embedding model
* **Groq API** — LLM inference
* **GPT-OSS 20B** — language model
* **PyPDF / PDF processing tools**
* **python-docx** — DOCX processing

## Project Structure

```text
docquery/
│
├── loaders/
│   ├── pdf_loader.py
│   └── docx_loader.py
│
├── pipelines/
│   └── rag_pipeline.py
│
├── app.py
├── main.py
├── router.py
├── llm.py
├── config.py
├── test_docx.py
├── .gitignore
└── README.md
```

## How It Works

### 1. Document Ingestion

The application accepts supported documents and extracts their textual content.

### 2. Chunking

The extracted text is divided into smaller chunks so that relevant portions can be retrieved for individual questions.

### 3. Embedding

Each chunk is converted into a vector representation using:

```text
mixedbread-ai/mxbai-embed-large-v1
```

### 4. Vector Storage

The embeddings and their associated document content are stored in ChromaDB.

### 5. Retrieval

When a user asks a question, the system performs semantic similarity search and retrieves the most relevant chunks.

The current configuration retrieves:

```text
Top K = 5
```

### 6. Answer Generation

The retrieved context is passed to the Groq LLM:

```text
openai/gpt-oss-20b
```

The model is instructed to answer using the provided document context.

If the retrieved context does not contain the required information, the application can respond that the information is not available in the document rather than relying on the model's general knowledge.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nevinjosephantony/docquery.git
cd docquery
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the required Python packages used by the project.

```bash
pip install -r requirements.txt
```

> If `requirements.txt` has not yet been added to the repository, install the project's dependencies manually for now. A pinned `requirements.txt` will be added separately.

### 4. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
MXBAI_API_KEY=your_mxbai_api_key
```

Do **not** commit the `.env` file to GitHub.

### 5. Run the application

```bash
python app.py
```

Follow the application's interface to upload a document and ask questions about it.

## Example

After uploading a document containing a research paper, users can ask:

```text
What are the main objectives of the proposed system?
```

The system retrieves relevant sections of the document and generates an answer based on that context.

For an unrelated question such as:

```text
What is a lion?
```

when the uploaded document contains no information about lions, the system is designed to indicate that the information is not available in the provided context.

## Configuration

The main configuration is located in `config.py`.

Important settings include:

```python
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

LLM_MODEL = "openai/gpt-oss-20b"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

TOP_K = 5
```

API keys are loaded from environment variables rather than being stored directly in the source code.

## Current Scope

This project currently focuses on document question answering using a basic RAG architecture.

Future improvements can include:

* More advanced document parsing
* Structure-aware chunking
* Improved retrieval strategies
* Re-ranking
* Better source citation
* Support for additional document formats
* Retrieval evaluation and benchmarking
* Improved handling of tables and images
* More robust deployment and configuration

## Security

API keys and local uploaded documents are excluded from version control through `.gitignore`.

Never commit credentials or private documents to the repository.

## License

This project is intended as a personal/academic project. No open-source license has currently been specified.
