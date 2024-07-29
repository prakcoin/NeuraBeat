import torch
import random as rand
import torchaudio
import torchaudio.transforms as T
from torchvision.transforms import v2
from model.model import ClassificationModel, EmbeddingModel

def load_model(model_path, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'classification':
        model = ClassificationModel()
    elif model_type == 'embedding':
        model = EmbeddingModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model

def preprocess(song, sr=16000, n_mels=128, n_fft=2048, hop_length=512, mean=6.5304, std=11.8924):
    chunk_duration = 3
    chunk_length = sr * chunk_duration

    song_waveform, w_sr = torchaudio.load(song)
    song_waveform = T.Resample(orig_freq=w_sr, new_freq=sr)(song_waveform)
    song_waveform = torch.mean(song_waveform, dim=0)
    song_waveform = song_waveform[0:chunk_length]
    song_waveform = song_waveform.unsqueeze(0)

    mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)(song_waveform)
    log_mel_spec =  T.AmplitudeToDB()(mel_spec)
    mel_spec_tensor = log_mel_spec.unsqueeze(0)
    mel_spec_tensor = v2.Compose([v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                                  v2.Normalize((mean,), (std,))])(mel_spec_tensor)

    return mel_spec_tensor