import torch
import random as rand
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import psycopg2
from flask import Flask, request, render_template
from torchvision.transforms import v2
from model.model import ClassificationModel, EmbeddingModel
from utils.db import insert_embedding, embedding_exists, retrieve_similar_embeddings

app = Flask(__name__)

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST')
)

class_names = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

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
    
    song_len = len(song_waveform)
    bound = rand.randint(0, song_len - chunk_length)
    song_waveform = song_waveform[bound:bound + chunk_length]

    song_waveform = song_waveform.unsqueeze(0)

    mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)(song_waveform)
    log_mel_spec =  T.AmplitudeToDB()(mel_spec)
    mel_spec_tensor = log_mel_spec.unsqueeze(0)
    mel_spec_tensor = v2.Compose([v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                                  v2.Normalize((mean,), (std,))])(mel_spec_tensor)

    return mel_spec_tensor

@app.route('/process_file', methods=['POST'])
def process_file():
    song = request.files['audio']
    action = request.form['action']
    image_tensor = preprocess(song)
    
    if action == 'Predict':
        classification_model = load_model('model/classification_model.pt', 'classification')
        output = classification_model(image_tensor)

        probabilities = F.softmax(output, dim=1)
        probabilities = probabilities.detach().numpy()[0]
        class_index = probabilities.argmax()

        predicted_class = class_names[class_index]
        probability = probabilities[class_index]

        class_probs = list(zip(class_names, probabilities))
        class_probs.sort(key=lambda x: x[1], reverse=True)

        return render_template('classify.html', class_probs=class_probs,
                           predicted_class=predicted_class, probability=probability)
    elif action == 'Embed':
        embedding_model = load_model('model/embedding_model.pt', 'embedding')
        embedding = embedding_model(image_tensor)
        embedding = embedding.flatten().detach().cpu().numpy().tolist()
        # if (not embedding_exists(conn, embedding)):
        #     insert_embedding(conn, embedding)
        
        embeddings_with_distances = retrieve_similar_embeddings(conn, embedding)

        return render_template('embed.html', embeddings_with_distances=embeddings_with_distances)

@app.route('/')  
def home():  
    return render_template('home.html')
