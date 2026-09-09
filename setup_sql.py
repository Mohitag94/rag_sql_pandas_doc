"""One-time script: create the SQL schema and tables in Neon."""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

CREATE_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS rag_app;
"""

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS rag_app.document_metadata (
	id SERIAL PRIMARY KEY,
	source TEXT UNIQUE NOT NULL,
	date_added TIMESTAMP DEFAULT NOW(),
	category TEXT
);
"""

CREATE_QUERY_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS rag_app.query_log (
	id SERIAL PRIMARY KEY,
	timestamp TIMESTAMP DEFAULT NOW(),
	query_text TEXT NOT NULL,
	answer TEXT,
	retrieved_chunk_ids TEXT,
	retrieved_chunk_texts TEXT,
	confidence_score FLOAT,
	latency_ms INTEGER,
	error TEXT
);
"""

CREATE_RAG_EVAL_TABLE = """
CREATE TABLE IF NOT EXISTS rag_app.rag_eval (
	id SERIAL PRIMARY KEY, 
	query_log_id INTEGER NOT NULL REFERENCES rag_app.query_log(id), 
	evaluated_at TIMESTAMP DEFAULT NOW(), 
	faithfulness_passing BOOLEAN, 
	faithfulness_feedback TEXT, 
	relevancy_passing BOOLEAN, 
	relevancy_feedback TEXT )
"""


def main():
    conn_string = os.getenv("DATABASE_URL")
    if not conn_string:
        raise ValueError("DATABASE_URL environment variable is missing!")

    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            # execute the schema creation first
            cur.execute(CREATE_SCHEMA)
            # create tables inside the schema
            cur.execute(CREATE_METADATA_TABLE)
            cur.execute(CREATE_QUERY_LOG_TABLE)
            cur.execute(CREATE_RAG_EVAL_TABLE)
        conn.commit()
    print("Schema 'rag_app' and both tables created (or already existed).")


if __name__ == "__main__":
    main()
