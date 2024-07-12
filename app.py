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
from utils.utils import preprocess, load_model

app = Flask(__name__)

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST')
)

class_names = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

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
