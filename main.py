from router import route_file
from pipelines.rag_pipeline import run_rag
from pipelines.sql_pipeline import (
    load_csv_to_sqlite,
    get_schema,
    schema_to_text,
    generate_sql,
    clean_sql,
    execute_sql,
    generate_answer
)

file_path = input("Enter file path: ")

pipeline = route_file(file_path)

if pipeline == "sql_pipeline":

    conn = load_csv_to_sqlite(
        file_path
    )

    schema = get_schema(
        conn
    )

    schema_text = schema_to_text(
        schema
    )

    print(schema_text)

    question = input(
        "\nEnter question: "
    )

    sql_query = generate_sql(
        question,
        schema_text
    )

    sql_query = clean_sql(
        sql_query
    )

    print(
        "\nGenerated SQL:"
    )

    print(
        sql_query
    )

    results = execute_sql(
        conn,
        sql_query
    )

    answer = generate_answer(
        question,
        sql_query,
        results
    )

    print(
        "\nANSWER"
    )

    print(
        "=" * 50
    )

    print(
        answer
    )

elif pipeline == "rag_pipeline":

    question = input(
        "\nEnter question: "
    )

    answer = run_rag(
        file_path,
        question
    )

    print(
        "\nANSWER"
    )

    print(
        "=" * 50
    )

    print(
        answer
    )

else:

    print(
        "Unsupported file type"
    )

def generate_answer(
    question,
    sql_query,
    results
):

    prompt = f"""
Question:
{question}

SQL Query:
{sql_query}

Query Results:
{results}

Answer the question in natural language.
"""

    return call_llm(
        prompt
    )