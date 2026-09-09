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
			INSERT INTO rag_app.query_log 
			(query_text, answer, retrieved_chunk_ids, 
			retrieved_chunk_texts, confidence_score, latency_ms, error)
			VALUES (%s, %s, %s, %s, %s, %s, %s)
			""",
            (
                result["query_text"],
                result["answer"],
                chunk_ids_str,
                chunk_texts_str,
                result["confidence_score"],
                result["latency_ms"],
                result["error"],
            ),
        )
        conn.commit()


def fetch_unevaluated(limit=15):
    """
    Fetch unevaluated rows from query_log oldest first, excluding
    rows where the response failed.

    Args:
                limit: max number of rows to be fetched.

    Return:
                a list of tuple: (id, query_test, question, retrieved_chuck_texts)
    """

    conn_string = os.getenv("DATABASE_URL")
    if not conn_string:
        raise ValueError("DATABASE_URL environment variable is missing!")

    with psycopg.connect(conn_string) as conn, conn.cursor() as cur:
        cur.execute(
            """
			SELECT q1.id, q1.query_text, q1.answer, q1.retrieved_chunk_texts 
			FROM rag_app.query_log q1
			WHERE q1.error IS NULL 
			AND NOT EXISTS (
				SELECT 1 FROM rag_app.rag_eval er WHERE er.query_log_id = q1.id
				)
			ORDER BY q1.id
			LIMIT %s
			""",
            (limit,),
        )
        return cur.fetchall()


def eval_logger(results: list[dict]):
    """
    Records evaluation per query in a batch.

    Args:
            result: list of dicts, each with keys query_log_id,
            faithfulness_passing, faithfulness_feedback, relevancy_passing,
            relevancy_feedback — the exact shape BatchEvaluator.run() returns.
    """

    if not results:
        return

    conn_string = os.getenv("DATABASE_URL")
    if not conn_string:
        raise ValueError("DATABASE_URL environment variable is missing!")

    with psycopg.connect(conn_string) as conn, conn.cursor() as cur:
        cur.executemany(
            """
			INSERT INTO rag_app.rag_eval 
				(query_log_id, 
				faithfulness_passing, 
				faithfulness_feedback,
				relevancy_passing, 
				relevancy_feedback) 
				VALUES (%(query_log_id)s, 
						%(faithfulness_passing)s, 
						%(faithfulness_feedback)s,
						%(relevancy_passing)s, 
						%(relevancy_feedback)s)
			""",
            results,
        )
        conn.commit()
