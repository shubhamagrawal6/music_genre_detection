# Imports for plots and system
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile as wav
import pandas as pd
import os
import numpy as np
import seaborn as sns

# Imports for ML models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import catboost as cb

# Imports for Training and accuracey
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,LearningRateScheduler
# import tensorflow.keras as keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import *

# Plot graph of the audio file
def plot_sound(path):
    plt.figure(figsize=(14, 5))
    data, sample_rate = librosa.load(path)
    print("length {}, sample-rate {}".format(data.shape, sample_rate))
    librosa.display.waveplot(data, sr=sample_rate)
    
    return data

# Plot spectrogram of the audio file
def plot_spectrogram(path):
    x, sr = librosa.load(path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

