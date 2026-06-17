import sys
import os
from sentence_transformers import CrossEncoder

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

def store_chunks(
    chunks,
    collection_name="documents"
):

    collection = client.get_or_create_collection(
        collection_name
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB"
    )

    return collection


def retrieve(
    query,
    collection_name="documents",
    top_k=3
):

    query_embedding = embed_texts(
        [query]
    )[0]

    collection = client.get_or_create_collection(
        collection_name
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k
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


def hybrid_retrieve(
    query,
    chunks,
    top_k=5
):

    vector_results = retrieve(
        query,
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

    return combined

def expand_context(results, chunks, window=1):

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

    return expanded
def rerank_chunks(
    query,
    expanded_chunks,
    top_k=5
):

    pairs = []

    for chunk in expanded_chunks:

        pairs.append(
            (
                query,
                chunk["text"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(
            expanded_chunks,
            scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk
        for chunk, score in ranked[:top_k]
    ]

if __name__ == "__main__":

    from loaders.pdf_loader import (
        ingest_pdf,
        chunk_document
    )

    doc = ingest_pdf(
        r"D:\chatbot\aonz\claude free Updated doc.pdf"
    )

    chunks = chunk_document(
        doc,
        "claude_free.pdf"
    )

    store_chunks(
        chunks
    )

    query = (
        "what are the free credits available?"
    )

    results = hybrid_retrieve(
        query,
        chunks
    )

    expanded = expand_context(
        results,
        chunks
    )

    reranked = rerank_chunks(
        query,
        expanded,
        top_k=5
    )
    from llm import call_llm

    context = "\n\n".join(
        chunk["text"]
        for chunk in reranked
    )

    prompt = f"""
Answer the question using ONLY the context below.

Question:
{query}

Context:
{context}

    Answer:
    """

    answer = call_llm(prompt)

    print("\nANSWER")
    print("=" * 50)
    print(answer)
 


