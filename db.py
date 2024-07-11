import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE

# Store: song name, embedding

def insert_embedding(conn, embedding_vector):
    sql = "INSERT INTO embeddings (vector) VALUES (%s);"
    cursor = conn.cursor()
    cursor.execute(sql, (embedding_vector,))
    conn.commit()
    cursor.close()

def embedding_exists(conn, embedding_vector):
    sql = "SELECT EXISTS(SELECT 1 FROM embeddings WHERE vector = %s);"
    cursor = conn.cursor()
    cursor.execute(sql, (embedding_vector,))
    exists = cursor.fetchone()[0]
    cursor.close()
    return exists

def retrieve_similar_embeddings(conn, embedding_vector):
    sql = "SELECT * FROM items ORDER BY embedding <-> '(%s)' LIMIT 5;"
    cursor = conn.cursor()
    cursor.execute(sql, (embedding_vector),)
    rows = cursor.fetchall()
    embeddings = [row[0] for row in rows]
    cursor.close()
    return embeddings