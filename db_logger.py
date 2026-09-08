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

    chunk_ids_str = (
        ",".join(result["retrieved_chunk_ids"])
        if result["retrieved_chunk_ids"]
        else None
    )
    with psycopg.connect(conn_string) as conn, conn.cursor() as cur:
        cur.execute(
            """
			INSERT INTO raq_app.query_log 
			(query_text, retrieved_chunk_ids, confidence_score, latency_ms, error)
			VALUES (%s, %s, %s, %s, %s)
			""",
            (
                result["query_text"],
                chunk_ids_str,
                result["confidence_score"],
                result["latency_ms"],
                result["error"],
            ),
        )
        conn.commit()
