
# %% Load
import streamlit as st
import time
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

st.set_page_config(layout="centered", page_icon="📁", page_title="Try_Your_Music")
st.sidebar.header('Try it with your music!')

device = torch.device('cpu')

# Define pretrained resnet from Torch Vision resnet 18
class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, dropout_rate=0.8):
        # get resnet super class
        super(ResNetEmbedding, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        # Change structure of first layer to take non RGB images, rest of params same as default
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        # Set the last fully connected to a set dimension "embedding_dim" instead of default 1000
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.resnet(x)
        return F.normalize(x, p=2, dim=1)

model = ResNetEmbedding()  # Make sure this matches the architecture you used
model.to(device)
#if torch.cuda.is_available():
#    model.cuda()
#model.load_state_dict(torch.load('resnet18_model_weights.pth'))
model.load_state_dict(torch.load('pages/resnet18_model_weights.pth', map_location=torch.device('cpu')))

def extract_embedding(model, audio_data_clip, sr=22050, use_model=True):
    y = audio_data_clip
    #y, sr = librosa.load(audio_data_clip, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Convert to tensor and move to the appropriate device
    mel_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    if use_model:
        # Get the embedding from the model
        with torch.no_grad():
            embedding = model(mel_tensor)
        
        # Normalize the embedding
        #embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    else:
        return mel_tensor


def compute_cosine_similarity(embedding1, embedding2):
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding1, embedding2)
    return cosine_sim.item()  # Convert to a Python float

def apply_model(file1, file2):
    audio1 = librosa.load(file1)[0]
    audio2 = librosa.load(file2)[0]
    emb1 = extract_embedding(model, audio1)
    emb2 = extract_embedding(model, audio2)
    return compute_cosine_similarity(emb1, emb2)

##########################################
 
st.title('🎶 Song Similarity')

st.markdown('Welcome! This page is for testing our model on user-provided clips. Upload two music clips to see their similarity score:')

# side_1, side_2 = st.columns(2)
# 
# with side_1:
#     song_file_1 = st.file_uploader('Song 1', type = ['mp3', 'wav'])
# 
# with side_2:
#     song_file_2 = st.file_uploader('Song 2', type = ['mp3', 'wav'])

song_file_1 = st.file_uploader('Song 1', type = ['mp3', 'wav'])

song_file_2 = st.file_uploader('Song 2', type = ['mp3', 'wav'])

results_container = st.empty()

st.divider()

st.markdown('Made for the Erd&#337;s Institute\'s Deep Learning Boot Camp, Summer 2024.')

if song_file_1 is not None and song_file_2 is not None:
    results_container.markdown('Processing...')
    time.sleep(.5)
    results_container.markdown('Processing....')
    time.sleep(.5)
    results_container.markdown('Processing.....')
    time.sleep(.5)
    result = apply_model(song_file_1, song_file_2)
    results_statement = '**The cosine similarity score is: ' + str(result) + '**  \n(A higher value means greater similarity, with a maximum of 1.)'
    results_container.markdown(results_statement)
else:
    results_container.markdown('The result will show here once you upload the files.')

