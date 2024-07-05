import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from flask import Flask, request, render_template
from torchvision.transforms import v2
from model.model import ClassificationModel

app = Flask(__name__)

model_path = 'model/classification_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()

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

def process_audio(song):
    song_waveform, w_sr = torchaudio.load(song)
    song_waveform = T.Resample(orig_freq=w_sr, new_freq=sr)(song_waveform)
    song_waveform = torch.mean(song_waveform, dim=0).unsqueeze(0)

    mel_spec = mel_spec_transform(song_waveform)
    log_mel_spec = log_mel_spec_transform(mel_spec)
    mel_spec_tensor = log_mel_spec.unsqueeze(0)
    mel_spec_tensor = image_transforms(mel_spec_tensor)

    return mel_spec_tensor

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image file
    song = request.files['audio']

    # Process image and make prediction
    image_tensor = process_audio(song)
    output = model(image_tensor)

    # Get class probabilities
    probabilities = F.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    # Get the index of the highest probability
    class_index = probabilities.argmax()

    # Get the predicted class and probability
    predicted_class = class_names[class_index]
    probability = probabilities[class_index]

    # Sort class probabilities in descending order
    class_probs = list(zip(class_names, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Render HTML page with prediction results
    return render_template('predict.html', class_probs=class_probs,
                           predicted_class=predicted_class, probability=probability)

@app.route('/')  
def home():  
    return render_template('home.html')
