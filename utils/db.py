import psycopg2
import os
import sys
import boto3
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST')
)

def create_table(conn):
    cur = conn.cursor()
    cur.execute("""
                CREATE TABLE song_embeddings (
                    id bigserial PRIMARY KEY,
                    song_name TEXT NOT NULL,
                    genre TEXT,
                    s3_url TEXT,
                    embedding vector(128)
                );
                """)

    cur.execute("""
                ALTER TABLE song_embeddings ADD CONSTRAINT unique_embedding UNIQUE (embedding);
                """)

    conn.commit()
    cur.close()

def delete_table(conn):
    cur = conn.cursor()
    cur.execute("""
                DROP TABLE song_embeddings;
                """)
    conn.commit()
    cur.close()

# TODO: NEED TO ADD S3 FUNCTIONALITY
def insert_embedding(conn, song_name, genre, s3_url, embedding):
    cur = conn.cursor()
    cur.execute("""
                INSERT INTO song_embeddings (song_name, genre, s3_url, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (embedding) DO NOTHING;
                """, (song_name, genre, s3_url, embedding))
    conn.commit()
    cur.close()

def embedding_exists(conn, embedding):
    cur = conn.cursor()
    embedding = '[' + ','.join(map(str, embedding)) + ']'
    cur.execute("""
                SELECT id
                FROM song_embeddings
                WHERE embedding <-> %s = 0.0;  -- Adjust the threshold as needed
                """, (embedding,))
    exists = cur.fetchone()
    cur.close()
    return exists

def retrieve_similar_embeddings(conn, embedding, s3_client, bucket):
    cur = conn.cursor()
    embedding = '[' + ','.join(map(str, embedding)) + ']'
    
    cur.execute("""
                SELECT song_name, genre, s3_url
                FROM song_embeddings
                WHERE (embedding <-> %s) = 0.0;
                """, (embedding,))

    song_name, genre, s3_url = cur.fetchone()

    object_name = s3_url.split(f"https://{bucket}.s3.amazonaws.com/")[-1]
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': object_name},
        ExpiresIn=3600
    )
    input_embedding = (song_name, genre, presigned_url)

    cur.execute("""
                SELECT song_name, genre, s3_url, (embedding <-> %s) AS distance
                FROM song_embeddings
                WHERE (embedding <-> %s) > 0.0
                ORDER BY embedding <-> %s
                LIMIT 5;
                """, (embedding, embedding, embedding))
    rows = cur.fetchall()

    embeddings_with_distances = []
    for row in rows:
        song_name, genre, s3_url, distance = row
        object_name = s3_url.split(f"https://{bucket}.s3.amazonaws.com/")[-1]
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': object_name},
            ExpiresIn=3600
        )
        embeddings_with_distances.append((song_name, genre, presigned_url, distance))
    
    cur.close()
    return embeddings_with_distances, input_embedding

def upload_to_s3(file_path, bucket_name, object_name):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    except Exception as e:
        print(f"Error uploading {file_path} to S3: {e}")
        return None