from router import route_file
from pipelines.rag_pipeline import run_rag, ingest_document, ask_question
from pipelines.sql_pipeline import (
    load_csv_to_sqlite,
    get_schema,
    schema_to_text,
    generate_sql,
    clean_sql,
    execute_sql,
    generate_answer
)

file_inputs = input("Enter file path(s), comma separated: ")
file_paths = [f.strip() for f in file_inputs.split(",")]

all_rag_chunks = []
sql_connections = []

for file_path in file_paths:
    pipeline = route_file(file_path)

    if pipeline == "rag_pipeline":
        chunks = ingest_document(file_path)
        all_rag_chunks.extend(chunks)

    elif pipeline == "sql_pipeline":
        conn = load_csv_to_sqlite(file_path)
        sql_connections.append((file_path, conn))

    else:
        print(f"Skipping unsupported file: {file_path}")

question = input("\nEnter question: ")

# RAG answer
if all_rag_chunks:
    # pass all sources as comma joined string
    all_sources = ",".join(
        list(set(c["metadata"]["source"] for c in all_rag_chunks))
    )
    answer = ask_question(question, all_rag_chunks, all_sources)
    print("\nANSWER (Documents)")
    print("=" * 50)
    print(answer)

# SQL answer
for file_path, conn in sql_connections:
    schema = get_schema(conn)
    schema_text = schema_to_text(schema)
    sql_query = generate_sql(question, schema_text)
    sql_query = clean_sql(sql_query)
    results = execute_sql(conn, sql_query)
    answer = generate_answer(question, sql_query, results)
    print(f"\nANSWER ({file_path})")
    print("=" * 50)
    print(answer)