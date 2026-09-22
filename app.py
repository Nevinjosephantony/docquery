import streamlit as st
import sys
import os
import sqlite3

# ── path setup so imports work from project root ──────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from router import route_file
from pipelines.rag_pipeline import ingest_document, ask_question
from pipelines.sql_pipeline import (
    load_csv_to_sqlite,
    get_schema,
    schema_to_text,
    generate_sql,
    clean_sql,
    execute_sql,
    generate_answer,
)
import chromadb
from config import VECTOR_DB_PATH

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocQuery",
    page_icon="📄",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────
def get_ingested_files():
    """Return list of unique source filenames already in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_or_create_collection("documents")
        results = collection.get()
        sources = list({m["source"] for m in results["metadatas"]}) if results["metadatas"] else []
        return sorted(sources)
    except Exception:
        return []


def save_uploaded_file(uploaded_file):
    """Save streamlit UploadedFile to a temp path, return path string."""
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def delete_file_from_chromadb(source_name):
    """Delete all chunks for a given source from ChromaDB."""
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection("documents")
    results = collection.get(where={"source": source_name})
    if results["ids"]:
        collection.delete(ids=results["ids"])


# ── session state defaults ────────────────────────────────────────────────────
if "sql_connections" not in st.session_state:
    st.session_state.sql_connections = {}   # {filename: conn}
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = {}        # {filename: [chunks]}
if "answers" not in st.session_state:
    st.session_state.answers = []        
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DocQuery")
    st.caption("Multi-format document intelligence")
    st.divider()

    # File uploader
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, CSV, XLSX",
        accept_multiple_files=True,
        type=["pdf", "docx", "csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚙️ Process Files", use_container_width=True):
            try:
                client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
                client.delete_collection("documents")
            except Exception:
                pass
            st.session_state.rag_chunks = {}
            st.session_state.sql_connections = {}
            for uf in uploaded_files:
                path = save_uploaded_file(uf)
                try:
                    pipeline = route_file(path)
                    if pipeline == "rag_pipeline":
                        with st.spinner(f"Ingesting {uf.name}..."):
                            chunks = ingest_document(path)
                            st.session_state.rag_chunks[uf.name] = chunks
                        st.success(f"✓ {uf.name} ({len(chunks)} chunks)")
                    elif pipeline == "sql_pipeline":
                        with st.spinner(f"Loading {uf.name}..."):
                            conn = load_csv_to_sqlite(path)
                            st.session_state.sql_connections[uf.name] = conn
                        st.success(f"✓ {uf.name} (SQL ready)")
                except Exception as e:
                    st.error(f"✗ {uf.name}: {e}")

    st.divider()

    # Already ingested files
    st.subheader("Knowledge Base")
    ingested = get_ingested_files()
    if ingested:
        for fname in ingested:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 `{fname}`")
            if col2.button("🗑", key=f"del_{fname}", help=f"Delete {fname}"):
                delete_file_from_chromadb(fname)
                if fname in st.session_state.rag_chunks:
                    del st.session_state.rag_chunks[fname]
                st.rerun()
    else:
        st.caption("No documents ingested yet.")

    st.divider()

    # Active SQL files this session
    if st.session_state.sql_connections:
        st.subheader("SQL Files (this session)")
        for fname in st.session_state.sql_connections:
            st.markdown(f"📊 `{fname}`")

# ── main area ─────────────────────────────────────────────────────────────────
st.header("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="What is this document about?",
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([1, 5])
ask_clicked = col_ask.button("Ask", type="primary", use_container_width=True)
if col_clear.button("Clear answers"):
    st.session_state.answers = []
    st.session_state.chat_history = []

if ask_clicked and question.strip():
    all_rag_chunks = []
    for chunks in st.session_state.rag_chunks.values():
        all_rag_chunks.extend(chunks)

    

    new_answers = []

    # RAG
    if all_rag_chunks:

        with st.spinner("Answering..."):

            all_sources = ",".join(
                set(
                    c["metadata"]["source"]
                    for c in all_rag_chunks
                )
            )

            result = ask_question(
                question,
                all_rag_chunks,
                all_sources,
                st.session_state.chat_history
            )

            new_answers.append({
                "source": "DocQuery",
                "type": "rag",
                "answer": result["answer"],
                "sources": result["sources"],
                "sql": None
            })
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": result["answer"]
                }
            )
    # SQL
    for fname, conn in st.session_state.sql_connections.items():
        schema = get_schema(conn)
        schema_text = schema_to_text(schema)
        with st.spinner(f"Generating SQL for {fname}..."):
            sql_query = clean_sql(generate_sql(question, schema_text))
        try:
            results = execute_sql(conn, sql_query)
            with st.spinner(f"Summarising results from {fname}..."):
                answer = generate_answer(question, sql_query, results)
            new_answers.append({"source": fname, "type": "sql", "answer": answer, "sql": sql_query})
        except Exception as e:
            new_answers.append({"source": fname, "type": "sql", "answer": f"SQL error: {e}", "sql": sql_query})

    if not new_answers:
        st.warning("No files loaded. Upload and process files first.")
    else:
        st.session_state.answers = new_answers

elif ask_clicked:
    st.warning("Please enter a question.")

# ── display answers ───────────────────────────────────────────────────────────
for item in st.session_state.answers:

    icon = "📄" if item["type"] == "rag" else "📊"

    with st.expander(
        f"{icon} {item['source']}",
        expanded=True
    ):

        st.markdown(
            item["answer"]
        )

        if item["type"] == "rag":

            st.divider()

            st.subheader(
                "Sources"
            )

            for source in item.get(
                "sources",
                []
            ):

                st.write(
                    source
                )

        if item["sql"]:

            st.divider()

            st.caption(
                "Generated SQL"
            )

            st.code(
                item["sql"],
                language="sql"
            )