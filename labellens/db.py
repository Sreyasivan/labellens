"""Database connection and schema setup."""
import psycopg
from pgvector.psycopg import register_vector

from labellens import config


def connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


def init_schema() -> None:
    # Plain connection: the vector type cannot be registered until the extension exists.
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute(config.SCHEMA_SQL.read_text())
        conn.commit()


if __name__ == "__main__":
    init_schema()
    print("schema ready")
