"""One-time script: create the two SQL tables in Neon."""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS document_metadata (
	id SERIAL PRIMARY KEY,
	source TEXT NOT NULL,
	date_added TIMESTAMP DEFAULT NOW(),
	category TEXT
);
"""

CREATE_QUERY_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS query_log (
	id SERIAL PRIMARY KEY,
	timestamp TIMESTAMP DEFAULT NOW(),
	query_text TEXT NOT NULL,
	retrieved_chunk_ids TEXT,
	confidence_score FLOAT,
	latency_ms INTEGER,
	error TEXT
);
"""


def main():
    conn_string = os.getenv("DATABASE_URL")
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_METADATA_TABLE)
            cur.execute(CREATE_QUERY_LOG_TABLE)
        conn.commit()
    print("Both tables created (or already existed).")


if __name__ == "__main__":
    main()
