# Imports for plots and system
import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile as wav
import pandas as pd
from pandas import MultiIndex, Int64Index
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

df = pd.read_csv('Data/features_3_sec.csv')

# df.head()

# spike_cols = [col for col in df.columns if 'mean' in col]
# corr = df[spike_cols].corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=np.bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(16, 11));

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 20)
# plt.xticks(fontsize = 10)
# plt.yticks(fontsize = 10)

# x = df[["label", "tempo"]]

# fig, ax = plt.subplots(figsize=(16, 8));
# sns.boxplot(x = "label", y = "tempo", data = x, palette = 'husl');

# plt.title('BPM Boxplot for Genres', fontsize = 20)
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 10);
# plt.xlabel("Genre", fontsize = 15)
# plt.ylabel("BPM", fontsize = 15)
# plt.savefig("BPM_Boxplot.png")

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

X = df.drop(['label','filename'],axis=1)
y = df['label'] 

cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

# new data frame with the new scaled data. 
X = pd.DataFrame(np_scaled, columns = cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

def model_assess(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    joblib.dump(model, title+".sav")
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')

# Naive Bayes
nb = GaussianNB()
model_assess(nb, "NaiveBayes")

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, "StochasticGradientDescent")

# KNN
knn = KNeighborsClassifier(n_neighbors=19)
model_assess(knn, "KNN")

# Decission trees
tree = DecisionTreeClassifier()
model_assess(tree, "DecissionTrees")

# Random Forest
rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(rforest, "RandomForest")

# Support Vector Machine
svm = SVC(decision_function_shape="ovo")
model_assess(svm, "SupportVectorMachine")

# Logistic Regression
lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_assess(lg, "LogisticRegression")

# Neural Nets
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
model_assess(nn, "NeuralNets")

# catboost
cbc = cb.CatBoostClassifier(verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
model_assess(cbc,"CatBoostClassifier")

# Cross Gradient Booster
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, "CrossGradientBooster")

# Cross Gradient Booster (Random Forest)
xgbrf = XGBRFClassifier(objective= 'multi:softmax')
model_assess(xgbrf, "CrossGradientBoosterRandomForest")

