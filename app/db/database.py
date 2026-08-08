import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

connection_pool = None


class DatabaseUnavailableError(Exception):
    pass


def init_pool():
    global connection_pool
    try:
        connection_pool = SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)
    except Exception as e:
        logging.error(f"Pool initialization error: {e}")
        connection_pool = None


init_pool()


@contextmanager
def get_db_connection():
    global connection_pool

    if connection_pool is None:
        init_pool()
        if connection_pool is None:
            raise DatabaseUnavailableError("The database is unavailable.")

    current_pool = connection_pool
    conn = None
    try:
        conn = current_pool.getconn()
        if conn.closed:
            raise psycopg2.OperationalError("Connection closed.")
        register_vector(conn)
        yield conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
        logging.error(f"Database connection loss: {e}")
        if conn:
            try:
                current_pool.putconn(conn, close=True)
            except Exception:
                pass
            conn = None

        init_pool()
        raise DatabaseUnavailableError("Temporary database failure. Please retry the request.")
    finally:
        if conn and current_pool:
            try:
                current_pool.putconn(conn)
            except Exception:
                pass
