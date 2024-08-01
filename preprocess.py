import os
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import shutil
import soundfile as sf
import torch
import h5py
from librosa.util import fix_length
from sklearn.model_selection import train_test_split
from collections import defaultdict

mp3_data_path = '/mnt/c/Users/User/Documents/NeuraBeat/fma_small/'

for root, dirs, files in os.walk(mp3_data_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path != mp3_data_path + file:
            shutil.move(file_path, mp3_data_path)

for root, dirs, _ in os.walk(mp3_data_path, topdown=False):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        if os.path.isdir(folder_path):
            os.rmdir(folder_path)

data = []
max_songs_per_genre = 990
target_sr = 16000
chunk_duration = 3
chunk_length = target_sr * chunk_duration
full_song_length = 27

for mp3_file in os.listdir(mp3_data_path):
    try:
        song_waveform, w_sr = torchaudio.load(os.path.join(mp3_data_path, mp3_file))
        song_waveform = T.Resample(orig_freq=w_sr, new_freq=target_sr)(song_waveform)
        song_waveform = torch.mean(song_waveform, dim=0)
        padded_audio = fix_length(song_waveform, size=target_sr * full_song_length)
        audio_chunk = padded_audio[0:chunk_length]
        data.append(audio_chunk)


    # audio, sr = librosa.load(os.path.join(mp3_data_path, mp3_file))
    # if (len(audio) / sr) < full_song_length:
    #     print(f"Skipped short file: {mp3_file}")
    #     continue
    # resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # padded_audio = fix_length(resampled_audio, size=target_sr * full_song_length)
    # audio_chunk = padded_audio[0:chunk_length]
    # data.append(audio_chunk)

    except Exception as e:
        print(f"Skipped corrupt file: {mp3_file}")

data = np.array(data)

# training_data, val_data = train_test_split(data,
#                                             test_size=0.2,
#                                             random_state=42)

# val_data, test_data = train_test_split(val_data,
#                                         test_size=0.5,
#                                         random_state=42)

# with h5py.File('train_data.h5', 'w') as f:
#     f.create_dataset('data', data=np.array(training_data))

# with h5py.File('val_data.h5', 'w') as f:
#     f.create_dataset('data', data=np.array(val_data))

# with h5py.File('test_data.h5', 'w') as f:
#     f.create_dataset('data', data=np.array(test_data))