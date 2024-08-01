from dotenv import load_dotenv
import torch
import os
import psycopg2
import boto3
from flask import Flask, request, render_template
from utils.db import insert_embedding, embedding_exists, retrieve_similar_embeddings
from utils.utils import preprocess, load_model

app = Flask(__name__)

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST')
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
    region_name=os.getenv('S3_REGION')
)

class_names = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

@app.route('/process_file', methods=['POST'])
def process_file():
    song = request.files['audio']
    image_tensor = preprocess(song)
    
    embedding_model = load_model('model/saved models/embedding_model.pt')
    with torch.no_grad():
        embedding = embedding_model(image_tensor)
    embedding = embedding.flatten().detach().cpu().numpy().tolist()
    # if (not embedding_exists(conn, embedding)):
    #     insert_embedding(conn, embedding)
    
    similar_embeddings, input_embedding = retrieve_similar_embeddings(conn, embedding, s3_client, os.getenv('S3_BUCKET'))
    return render_template('embed.html', embedding=input_embedding, similar_embeddings=similar_embeddings)

@app.route('/')  
def home():  
    return render_template('home.html')
