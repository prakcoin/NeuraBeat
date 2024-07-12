import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE
from model.model import EmbeddingModel
import torchaudio.transforms as T
from torchvision.transforms import v2
from librosa.util import fix_length
import librosa
import torch
import csv
import os
import numpy as np
from collections import defaultdict

# mp3_data_path = '/mnt/c/Users/User/Documents/NeuraBeat/fma_small/'
# csv_path = '/mnt/c/Users/User/Documents/NeuraBeat/metadata/fma_metadata/tracks.csv'
# embedding_model_path = 'model/embedding_model.pt'

def insert_embedding(conn, song_name, genre, embedding_vector):
    cursor = conn.cursor()
    cursor.execute("""
                INSERT INTO song_embeddings (song_name, genre, embedding)
                VALUES (%s, %s, %s)
            """, (song_name, genre, embedding_vector))
    conn.commit()
    cursor.close()

def embedding_exists(conn, embedding_vector):
    embedding_array = '[' + ','.join(map(str, embedding_vector)) + ']'
    
    sql = "SELECT EXISTS(SELECT 1 FROM song_embeddings WHERE embedding = %s);"
    
    cursor = conn.cursor()
    cursor.execute(sql, (embedding_array,))
    exists = cursor.fetchone()[0]
    cursor.close()
    return exists

def retrieve_similar_embeddings(conn, embedding_vector):
    embedding_array = '[' + ','.join(map(str, embedding_vector)) + ']'
    
    sql = f"SELECT * FROM song_embeddings ORDER BY embedding <-> '{embedding_array}' LIMIT 5;"
    
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    embeddings = [(row[1], row[2]) for row in rows]
    cursor.close()
    return embeddings

def create_file_genre_map(mp3_data_path, csv_path):
    file_genre_map = {}  # Dictionary to store file-genre mapping
    track_ids = [file_name.split('.')[0].lstrip('0') for file_name in os.listdir(mp3_data_path) if file_name.endswith('.mp3')]

    # Read CSV file and create file-genre mapping
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) # Skip headers
        next(csvreader)
        next(csvreader)
        for row in csvreader:
            if row[0] in track_ids:
                genre = row[40]
                file_genre_map[row[0]] = genre
    
    return file_genre_map

def insert_all_embeddings(model, mp3_data_path, csv_path, file_genre_map, conn):
    cur = conn.cursor()
    genre_counts = defaultdict(int)
    max_songs_per_genre = 990
    target_sr = 16000
    n_mels=128
    n_fft=2048
    hop_length=512
    mean=6.5304
    std=11.8924

    chunk_duration = 3
    full_song_length = 27
    num_chunks = full_song_length // chunk_duration

    with open(csv_path, 'r') as csvfile:
        for mp3_file in os.listdir(mp3_data_path):
            track_id = mp3_file.split('.')[0].lstrip('0')
            genre = file_genre_map[track_id]
            if genre_counts[genre] >= max_songs_per_genre:
                continue

            try:    
                audio, sr = librosa.load(os.path.join(mp3_data_path, mp3_file))
                if (len(audio) / sr) < full_song_length:
                    print(f"Skipped short file: {mp3_file}")
                    continue
                resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                padded_audio = fix_length(resampled_audio, size=target_sr * full_song_length)

                chunk_length = target_sr * chunk_duration
                for i in range(num_chunks):
                    start_sample = i * chunk_length
                    end_sample = start_sample + chunk_length
                    if end_sample > len(padded_audio):
                        break
                    audio_chunk = torch.tensor(padded_audio[start_sample:end_sample]).unsqueeze(0)

                    mel_spec = T.MelSpectrogram(sample_rate=target_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)(audio_chunk)
                    log_mel_spec = T.AmplitudeToDB()(mel_spec)
                    mel_spec_tensor = log_mel_spec.unsqueeze(0)
                    mel_spec_tensor = v2.Compose([v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                                                  v2.Normalize((mean,), (std,))])(mel_spec_tensor)

                    with torch.no_grad():
                        embedding = model(mel_spec_tensor)
                    embedding = embedding.flatten().detach().cpu().numpy().tolist()

                    song_name = track_id + "_c" + str(i)

                    cur.execute("""
                        INSERT INTO song_embeddings (song_name, genre, embedding)
                        VALUES (%s, %s, %s)
                    """, (song_name, genre, embedding))
                    conn.commit()
                    print("Inserted track", song_name)

                genre_counts[genre] += 1

            except Exception as e:
                print(e)
                print(f"Skipped corrupt file: {mp3_file}")
    
    cur.close()
    conn.close()