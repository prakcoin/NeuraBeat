import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from flask import Flask, request, render_template
from torchvision.transforms import v2
from model.model import ClassificationModel, EmbeddingModel

app = Flask(__name__)

classification_model_path = 'model/classification_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = ClassificationModel()
classification_model.load_state_dict(torch.load(classification_model_path, map_location=torch.device('cpu')))
classification_model.to(device)
classification_model.eval()

embedding_model_path = 'model/embedding_model.pt'
embedding_model = EmbeddingModel()
embedding_model.load_state_dict(torch.load(embedding_model_path, map_location=torch.device('cpu')))
embedding_model.to(device)
embedding_model.eval()

class_names = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

sr=16000
n_mels=128
n_fft=2048
hop_length=512
mean=6.5304
std=11.8924

mel_spec_transform = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
log_mel_spec_transform = T.AmplitudeToDB()

image_transforms = v2.Compose([
                            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                            v2.Normalize((mean,), (std,))
                            ])

def preprocess(song):
    song_waveform, w_sr = torchaudio.load(song)
    song_waveform = T.Resample(orig_freq=w_sr, new_freq=sr)(song_waveform)
    song_waveform = torch.mean(song_waveform, dim=0).unsqueeze(0)

    mel_spec = mel_spec_transform(song_waveform)
    log_mel_spec = log_mel_spec_transform(mel_spec)
    mel_spec_tensor = log_mel_spec.unsqueeze(0)
    mel_spec_tensor = image_transforms(mel_spec_tensor)

    return mel_spec_tensor

@app.route('/process_file', methods=['POST'])
def process_file():
    song = request.files['audio']
    action = request.form['action']
    image_tensor = preprocess(song)
    
    if action == 'Predict':
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
        embedding = embedding_model(image_tensor)
        return render_template('embed.html')



@app.route('/')  
def home():  
    return render_template('home.html')
