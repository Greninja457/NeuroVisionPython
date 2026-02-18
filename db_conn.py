import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="NeuroVectorDB",
        user="postgres",
        password="GEC",
        host="localhost",
        port=5433
    )

def get_conn_cursor():
    conn = get_connection()
    return conn, conn.cursor()
