import sys
import os
from sentence_transformers import CrossEncoder
from pathlib import Path
from llm import call_llm

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import chromadb
from mixedbread import Mixedbread
from rank_bm25 import BM25Okapi

from config import (
    MXBAI_API_KEY,
    VECTOR_DB_PATH
)

mxbai = Mixedbread(
    api_key=MXBAI_API_KEY
)

client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

mxbai = Mixedbread(api_key=MXBAI_API_KEY)

def embed_texts(texts):
    result = mxbai.embed(
        model="mixedbread-ai/mxbai-embed-large-v1",
        input=texts
    )
    return [item.embedding for item in result.data]

def store_chunks(chunks, collection_name="documents"):
    # Filter out empty chunks before embedding
    chunks = [c for c in chunks if c["text"].strip()]
    
    if not chunks:
        print("No valid chunks to store")
        return None
    
    collection = client.get_or_create_collection(collection_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection

def retrieve(
    query,
    source_file,
    collection_name="documents",
    top_k=10
):
    """
    FIX: source_file is now actually used to filter the ChromaDB query.
    Previously this parameter was accepted but never applied, so vector
    search would pull chunks from every document ever ingested into the
    collection, not just the documents loaded in the current session.
    """

    query_embedding = embed_texts(
        [query]
    )[0]

    collection = client.get_or_create_collection(
        collection_name
    )

    # source_file may be a single filename or a comma-joined list of
    # filenames (as passed from app.py when multiple files are loaded)
    sources = [s.strip() for s in source_file.split(",") if s.strip()]

    where_filter = None
    if len(sources) == 1:
        where_filter = {"source": sources[0]}
    elif len(sources) > 1:
        where_filter = {"source": {"$in": sources}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    chunks = results["documents"][0]

    metadatas = results["metadatas"][0]

    return list(
        zip(
            chunks,
            metadatas
        )
    )


def bm25_retrieve(
    query,
    chunks,
    top_k=3
):

    tokenized_chunks = [

        chunk["text"]
        .lower()
        .split()

        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    tokenized_query = (
        query
        .lower()
        .split()
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked = sorted(
        zip(
            chunks,
            scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]

def keyword_retrieve(query, chunks, top_k=5):
    keywords = [
        word.lower() for word in query.split()
        if len(word) > 3
    ]

    scored = []
    for chunk in chunks:
        text = chunk["text"].lower()
        matches = sum(1 for kw in keywords if kw in text)
        if matches > 0:
            scored.append((chunk, matches))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def hybrid_retrieve(
    query,
    chunks,
    source_file,
    top_k=10
):

    vector_results = retrieve(
    query,
    source_file,
    top_k=top_k
)

    bm25_results = bm25_retrieve(
        query,
        chunks,
        top_k=top_k
    )

    combined = []

    seen = set()

    # Vector results
    for text, meta in vector_results:

        if text not in seen:

            combined.append(
                (
                    text,
                    meta
                )
            )

            seen.add(text)

    keyword_results = keyword_retrieve(query, chunks, top_k=top_k)
    for chunk, score in keyword_results:
        text = chunk["text"]
        if text not in seen:
            combined.append((text, chunk["metadata"]))
            seen.add(text)
    # BM25 results
    for chunk, score in bm25_results:

        text = chunk["text"]

        if text not in seen:

            combined.append(
                (
                    text,
                    chunk["metadata"]
                )
            )

            seen.add(text)
    print(f"Combined pool: {len(combined)} chunks")
    for text, meta in combined:
        print(f"  Source: {meta['source']} | Text: {text[:40]}")
    return combined

""" def expand_context(results, chunks, window=1):

    expanded = []
    seen = set()

    for text, meta in results:

        idx = meta["chunk_index"]

        start = max(0, idx - window)
        end = min(len(chunks), idx + window + 1)

        for i in range(start, end):

            chunk = chunks[i]

            if chunk["chunk_id"] not in seen:

                expanded.append(chunk)

                seen.add(
                    chunk["chunk_id"]
                )

    return expanded """
def rerank_chunks(query, expanded_chunks, top_k=5):
    pairs = [(query, chunk["text"]) for chunk in expanded_chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(expanded_chunks, scores), key=lambda x: x[1], reverse=True)
    for chunk, score in ranked[:top_k]:
        print(f"Score: {score:.3f} | Source: {chunk['metadata']['source']} | Text: {chunk['text'][:50]}")
    return [chunk for chunk, score in ranked[:top_k]]


def is_already_ingested(source_file, collection_name="documents"):
    collection = client.get_or_create_collection(collection_name)
    results = collection.get(where={"source": source_file})
    return len(results["ids"]) > 0

def ingest_document(file_path):
    source = os.path.basename(file_path)
    ext = Path(file_path).suffix.lower()

    if is_already_ingested(source):
        print(f"Already ingested: {source}, loading from ChromaDB")
        collection = client.get_or_create_collection("documents")
        results = collection.get(where={"source": source})
        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append({
                "chunk_id": results["ids"][i],
                "text": doc,
                "metadata": results["metadatas"][i]
            })
        return chunks

    if ext == ".pdf":
        from loaders.pdf_loader import ingest_pdf, chunk_document
        doc = ingest_pdf(file_path)
        chunks = chunk_document(doc, source)

    elif ext == ".docx":
        from loaders.docx_loader import ingest_docx, chunk_docx
        doc = ingest_docx(file_path)
        chunks = chunk_docx(doc, source)

    else:
        raise ValueError(f"Unsupported RAG file type: {ext}")

    store_chunks(chunks)
    return chunks

def ask_question(
    query,
    chunks,
    source_file,
    chat_history=None
):

    results = hybrid_retrieve(query, chunks, source_file)
# convert results to chunk format for reranker
    result_chunks = [{"chunk_id": f"r_{i}", "text": text, "metadata": meta}
                 for i, (text, meta) in enumerate(results)]
    reranked = rerank_chunks(query, result_chunks, top_k=5)
    sources = []

    sources = []

    for chunk in reranked:

        source = chunk["metadata"]["source"]
        section = chunk["metadata"]["section"]

        sources.append(
            f"{source} -> {section}"
        )

        sources = list(dict.fromkeys(sources))
    context = "\n\n".join(chunk["text"] for chunk in reranked)
    history = ""

    if chat_history:

        for msg in chat_history[-5:]:

            history += (
                f"User: {msg['question']}\n"
                f"Assistant: {msg['answer']}\n\n"
            )
    prompt = f"""
You are a document assistant.

Use the retrieved context to answer.

You may also use the previous conversation
to understand references such as
"it", "that", "the previous topic".

Do NOT use any knowledge that is
not present in the retrieved context.

Conversation History:

{history}

Context:

{context}

Question:

{query}

Answer:
"""
    print("\n--- PROMPT SENT TO LLM ---")
    print(prompt)
    print("--- END PROMPT ---\n")
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "sources": sources
    }


def run_rag(file_path, query):
    chunks = ingest_document(file_path)
    return ask_question(query, chunks, os.path.basename(file_path))
