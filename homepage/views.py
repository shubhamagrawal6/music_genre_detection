import os
import pickle
from django.shortcuts import render
import librosa
import numpy as np
from sklearn import feature_extraction

# Create your views here.

def find_audio():
    path = "C:/Users/agraw/Desktop/Projects/music_genre_detection/media/"

    mx = ""

    for file in os.listdir(path):
        x, _ = file.split(".")
        mx = max(mx, x)

    return path + mx + ".wav"

# Function to add features to the list and keep track of feature extraction
def addToList(arr, name, value):
    print(f"{name}: {value}")
    arr.append(value)

# Function to extract features
def extract_features(path):
    y, sr = librosa.load(path, duration=3, offset=10)

    arr = []
    
    # mfcc mean and variance values from 1 to 20
    for i in range(1, 21):
        mfcc = librosa.feature.mfcc(y=y, sr=sr)[i-1]
        addToList(arr, f"mfcc{i}_mean", np.mean(mfcc))                
        addToList(arr, f"mfcc{i}_var", np.var(mfcc))

    # chroma_stft mean and variance values
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    addToList(arr, "chroma_stft_mean", np.mean(chroma_stft))
    addToList(arr, "chroma_stft_var", np.var(chroma_stft))

    # rms mean and variance values
    rms = librosa.feature.rms(y=y)
    addToList(arr, "rms_mean", np.mean(rms))
    addToList(arr, "rms_var", np.var(rms))

    # spectral_centroid mean and variance values
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    addToList(arr, "spectral_centroid_mean", np.mean(spectral_centroid))
    addToList(arr, "spectral_centroid_var", np.var(spectral_centroid))

    # spectral_bandwidth mean and variance values
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    addToList(arr, "spectral_bandwidth_mean", np.mean(spectral_bandwidth))
    addToList(arr, "spectral_bandwidth_var", np.var(spectral_bandwidth))

    # spectral_rolloff mean and variance values
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    addToList(arr, "spectral_rolloff_mean", np.mean(spectral_rolloff))
    addToList(arr, "spectral_rolloff_var", np.var(spectral_rolloff))

    # zero_crossing_rate mean and variance values
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    addToList(arr, "zero_crossing_rate_mean", np.mean(zero_crossing_rate))
    addToList(arr, "zero_crossing_rate_var", np.var(zero_crossing_rate))

    # harmony mean and variance values
    harmony = librosa.effects.harmonic(y=y)
    addToList(arr, "harmony_mean", np.mean(harmony))
    addToList(arr, "harmony_var", np.var(harmony))

    # tempo mean and variance values
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    addToList(arr, "tempo", tempo)

    # Loading the scaler saved during model training
    with open('C:/Users/agraw/Desktop/Projects/music_genre_detection/homepage/minmax.pickle', 'rb') as handle:
        minmax = pickle.load(handle)

    # Normalizing the features using the scaler
    arr = minmax.transform([arr])

    return arr


def predict(arr):
    # Labels for prediction
    label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    with open('C:/Users/agraw/Desktop/Projects/music_genre_detection/homepage/CatBoostClassifier.pickle', 'rb') as handle:
        model = pickle.load(handle)

    preds = model.predict(arr)
    return label[preds[0][0]]

def results(request):
    genre = predict(extract_features(find_audio()))

    return render(request, "results.html", {"genre": genre, "model": "Cat Boost Classifier", "accuracy": "89.54" })