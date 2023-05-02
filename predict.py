import numpy as np
import librosa
import os
import glob
import warnings
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparseCoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import SparseCoder
# train and prediction based on my own dataset
import re

# MFCC
def extract_features(audio_file, mfcc=13):
    signal, sample_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=mfcc)
    return mfcc.mean(axis=1)

def extract_name(file_name):
    # The regular expression pattern below matches any sequence of letters (a-z or A-Z)
    pattern = r'./.*/[a-zA-Z]+'
    match = re.search(pattern, file_name)
    parts = match.group(0).split('/')
    # Get the string after the second '/'
    string_after_second_slash = parts[2]
    
    if match:
        return string_after_second_slash
    else:
        return None

def load_dataset(dir):
    # Load dataset
    data_dir = dir
    all_speakers = []
    all_speakers.extend(os.listdir(data_dir))
    X, y = [], []
    audio_files = glob.glob(os.path.join(data_dir, '*.wav'))
    audio_files +=  glob.glob(os.path.join(data_dir, '*.WAV'))

    for audio_file in audio_files:
        features = extract_features(audio_file,120)
        X.append(features)
        speaker = extract_name(audio_file)
        y.append(speaker)
        
    X, y = np.array(X), np.array(y)
    return X, y

X_train, y_train = load_dataset("./train")
X_test, y_test = load_dataset("./test")
dictionary = X_train.T

# print('Number of speakers:', len(np.unique(y_train)))
# print('Speakers:', np.unique(y_train))
# # improve robustness        
# identity_matrix = np.identity(X_train.shape[1])
# # print("Dictionary dimensions:", dictionary.shape)
# A_matrix = np.hstack((dictionary, identity_matrix))

# Sparse representation        
# Classification with Identity Matrix
coder = SparseCoder(dictionary.T, transform_algorithm='lasso_lars', transform_alpha=1.)
y_pred = []
for i, y in enumerate(X_test):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        x_est = coder.transform(y.reshape(1, -1))[0, :] # pinv(A)*y, estimated x

    residual = [
        np.linalg.norm(y - np.dot(
            dictionary[:, y_train == speaker][:, :len(X_train.T)], 
            x_est[:dictionary.shape[1]][y_train == speaker]))
        for speaker in np.unique(y_train)
    ]
    print('Number of speakers:', len(np.unique(y_train)))
    print(f"the residual of each speaker:\n{list(zip(np.unique(y_train),residual))}")
    y_pred.append(np.unique(y_train)[np.argmin(residual)])

print(f"The speaker is {y_pred[0].capitalize()}")