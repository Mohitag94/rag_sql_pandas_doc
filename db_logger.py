"""
For every question asked, the response along with it's metadata
are recorded into Neon raq_app's query_table.
"""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def query_logger(result: dict):
    """
    Records response into the table per question.

    Args:
            result: dict with keys query_text, retrieved_chunk_ids,
                    confidence_score, latency_ms, error (answer is ignored —
                    not stored in query_log).
    """
    conn_string = os.getenv("DATABASE_URL")
    if not conn_string:
        raise ValueError("DATABASE_URL environment variable is missing!")

    # join the ids to store in sql - dont support list
    chunk_ids_str = (
        ",".join(result["retrieved_chunk_ids"])
        if result["retrieved_chunk_ids"]
        else None
    )

    # join the text to store in sql - delimiter "\n--CHUNK--\n"
    chunk_texts_str = (
        "\n--CHUNK--\n".join(result["retrieved_chunk_texts"])
        if result["retrieved_chunk_texts"]
        else None
    )
    with psycopg.connect(conn_string) as conn, conn.cursor() as cur:
        cur.execute(
            """
			INSERT INTO raq_app.query_log 
			(query_text, answer, retrieved_chunk_ids, retrieved_chunk_text, confidence_score, latency_ms, error)
			VALUES (%s, %s, %s, %s, %s, %s, %s)
			""",
            (
                result["query_text"],
                result["answer"].response,
                chunk_ids_str,
                chunk_texts_str,
                result["confidence_score"],
                result["latency_ms"],
                result["error"],
            ),
        )
        conn.commit()
