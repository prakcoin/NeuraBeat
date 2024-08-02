import psycopg2
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from model.model import EmbeddingModel
import torchaudio.transforms as T
from torchvision.transforms import v2
from librosa.util import fix_length
import torch
import csv
import boto3
import torchaudio
from collections import defaultdict

mp3_data_path = '/mnt/c/Users/User/Documents/NeuraBeat/fma_small/'
csv_path = '/mnt/c/Users/User/Documents/NeuraBeat/metadata/fma_metadata/tracks.csv'
embedding_model_path = '../model/saved models/embedding_model.pt'
bucket_name = 'neurabeat'

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

def create_file_genre_map(mp3_data_path, csv_path):
    file_genre_map = {} 
    track_ids = [file_name.split('.')[0].lstrip('0') for file_name in os.listdir(mp3_data_path) if file_name.endswith('.mp3')]

    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) 
        next(csvreader)
        next(csvreader)
        for row in csvreader:
            if row[0] in track_ids:
                genre = row[40]
                file_genre_map[row[0]] = genre
    
    return file_genre_map

def insert_all_embeddings(model_path, mp3_data_path, file_genre_map, bucket_name, conn):
    cur = conn.cursor()
    genre_counts = defaultdict(int)
    max_songs_per_genre = 990
    target_sr = 16000
    n_mels=128
    n_fft=2048
    hop_length=512
    mean=6.5226
    std=10.4655
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    chunk_duration = 3
    full_song_length = 27
    num_chunks = full_song_length // chunk_duration

    for mp3_file in os.listdir(mp3_data_path):
        track_id = mp3_file.split('.')[0].lstrip('0')
        genre = file_genre_map[track_id]
        if genre_counts[genre] >= max_songs_per_genre:
            continue

        try:    
            audio, sr = torchaudio.load(os.path.join(mp3_data_path, mp3_file))
            resampled_audio = T.Resample(orig_freq=sr, new_freq=target_sr)(audio)
            resampled_audio = torch.mean(resampled_audio, dim=0)
            if (len(resampled_audio) / target_sr) < full_song_length:
                print(f"Skipped short file: {mp3_file}")
                continue
            padded_audio = fix_length(resampled_audio, size=target_sr * full_song_length)

            orig_chunk_length = sr * chunk_duration
            chunk_length = target_sr * chunk_duration
            for i in range(num_chunks):
                start_sample = i * chunk_length
                end_sample = start_sample + chunk_length

                orig_start_sample = i * orig_chunk_length
                orig_end_sample = orig_start_sample + orig_chunk_length 
                if end_sample > len(padded_audio):
                    break
                orig_audio_chunk = audio[:, orig_start_sample:orig_end_sample].clone().detach()
                audio_chunk = padded_audio[start_sample:end_sample].clone().detach().unsqueeze(0)

                song_name = track_id + "_c" + str(i)
                chunk_filename = f"{song_name}.wav"
                chunk_filepath = os.path.join('/mnt/c/Users/User/Documents/NeuraBeat/fma_small chunks/', chunk_filename)
                torchaudio.save(chunk_filepath, orig_audio_chunk, sr)

                s3_url = upload_to_s3(chunk_filepath, bucket_name, chunk_filename)
                if s3_url is None:
                    print(f"Failed to upload {chunk_filename} to S3.")
                    continue

                mel_spec = T.MelSpectrogram(sample_rate=target_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)(audio_chunk)
                log_mel_spec = T.AmplitudeToDB()(mel_spec)
                mel_spec_tensor = log_mel_spec.unsqueeze(0)
                mel_spec_tensor = v2.Compose([v2.Resize((64, 47)),
                                                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                                                v2.Normalize((mean,), (std,))])(mel_spec_tensor)

                with torch.no_grad():
                    embedding = model(mel_spec_tensor)
                embedding = embedding.flatten().detach().cpu().numpy().tolist()

                cur.execute("""
                    INSERT INTO song_embeddings (song_name, genre, s3_url, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (embedding) DO NOTHING;
                """, (song_name, genre, s3_url, embedding))
                conn.commit()
                if cur.rowcount == 0:
                    print("Skipped: Embedding already exists in the database.")
                else:
                    print("Inserted track", song_name)

            genre_counts[genre] += 1

        except Exception as e:
            print(e)
            print(f"Skipped corrupt file: {mp3_file}")

    cur.close()
    conn.close()