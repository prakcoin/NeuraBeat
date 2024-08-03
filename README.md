# NeuraBeat: Sonic Similarity Search through Deep Metric Learning
This project implements an end-to-end song similarity system. A neural network was trained using triplet loss and semi-hard triplet mining to generate embeddings for songs. These embeddings are then stored in a PostgreSQL database using pgvector for fast and accurate similarity search. A user-friendly web interface built with Flask is used for interacting with the neural network, allowing users to upload files for similarity search. The database retrieves the song name, genre, distance from the input song. Additionally, song files are stored in an AWS S3 bucket and retrieved using presigned URLs generated from their database names.

![Architecture Diagram](docs/NeuraBeat%20Architecture%20Diagram.png)

## Model Architecture
The below diagram was generated using the torchview library:

![Architecture Diagram](docs/NeuraBeat%20Model%20Diagram.png)

## Training
All training was conducted on Google Colab using PyTorch v2.3.1+cu121. The songs were converted to log scale mel spectrograms, resized, converted to tensors, and normalized. To maximize training results, a batch size of 960 (120 images per class using a balanced batch sampler) was used, along with a margin value of 0.1. This was the primary contributor to learning, as when the batch size increased and the margin value decreased, the loss and average mined triplets decreased. Triplet loss with Euclidean distance and regularized embeddings was employed. The model was trained for 30 epochs using the AdamW optimizer with a learning rate of 0.001 and a weight decay value of 0.01. Cosine annealing was utilized to decay the learning rate, along with AutoClip for gradient clipping. Notebooks for training and data preprocessing are attached in the model/train/ directory.

Below are training graphs for the current model, showing average loss over time and average mined triplets over time respectively:
![Loss Graph](docs/graphs/Loss%20Graph%20(0.1%20Margin).png)
![Mined Triplets Graph](docs/graphs/Mined%20Triplets%20Graph%20(0.1%20Margin).png)

## Setup
You can set up the environment using conda + pip or venv + pip, Python version 3.10.12 is required.

To install the packages run:
```
pip install -r requirements.txt
```

## Running the Code
To start the Flask app run:
```
flask --app app.py run
```

## Citations
This project relies on the following repositories:

- [PyTorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [Audiomentations](https://github.com/iver56/audiomentations)
- [AutoClip](https://github.com/pseeth/autoclip)
- [SeparableConv-Torch](https://github.com/reshalfahsi/separableconv-torch)
- [torchview](https://github.com/mert-kurttutan/torchview)