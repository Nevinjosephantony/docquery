import pandas as pd
import sqlite3

from llm import call_llm


def load_csv_to_sqlite(filepath):
    df = pd.read_csv(filepath)

    conn = sqlite3.connect(":memory:")

    df.to_sql(
        "data",
        conn,
        index=False,
        if_exists="replace"
    )

    return conn


def get_schema(conn):
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(data)")

    return cursor.fetchall()


def schema_to_text(schema):
    text = "Table: data\n\nColumns:\n"

    for column in schema:
        text += f"- {column[1]} ({column[2]})\n"

    return text


def generate_sql(question, schema_text):

    prompt = f"""
You are a SQLite expert.

Schema:

{schema_text}

Question:
{question}

Return ONLY a valid SQLite query.
No explanation.
"""

    return call_llm(prompt)


def clean_sql(sql_query):

    sql_query = sql_query.replace("```sql", "")
    sql_query = sql_query.replace("```", "")

    return sql_query.strip()


def execute_sql(conn, sql_query):

    cursor = conn.cursor()

    cursor.execute(sql_query)

    return cursor.fetchall()

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

    return call_llm(prompt)
