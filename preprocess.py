import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import warnings
from sklearn.model_selection import train_test_split
import os
import sys
import shutil
import time
import pywt
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback,ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, LayerNormalization
from keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten 
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.regularizers import L2
import math
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense
from tensorflow.keras.layers import multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import MaxNorm 
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import HeNormal, HeUniform
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Set logging level to suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
tf.get_logger().setLevel('ERROR')  

# Optional: Disable XLA if not needed
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import random
# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Ensure deterministic operations

def load_BCI2a_data(data_path, subject, training, all_trials = True):    
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48     
    window_Length = 7*250 
    
    # Define MI trial window 
    fs = 250          # sampling rate
    t1 = int(2*fs)  # start time_point
    t2 = int(6*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject+1)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject+1)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1        
    

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return

def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test
# Function to compute CWT
def batch_cwt(batch_signals, frequencies, sampling_frequency, normalization='zscore'):
    """
    Compute the Continuous Wavelet Transform (CWT) for a batch of signals using TensorFlow.

    Args:
        batch_signals (tf.Tensor): Input batch of signals with shape [batch_size, channels, signal_length].
        frequencies (np.array): Array of frequencies to use for the CWT.
        sampling_frequency (int): Sampling frequency of the signals.
        normalization (str): Type of normalization ('zscore' or 'minmax').

    Returns:
        tf.Tensor: Tensor containing the normalized CWT coefficients for the input batch of signals.
        Shape: (batch_size, frequencies, signal_length, channels)
    """
    import pywt
    batch_size, channels, signal_length = batch_signals.shape
    batch_signals_np = batch_signals.numpy()
    cwt_batch = []

    for i in range(batch_size):
        cwt_channels = []
        for ch in range(channels):
            signal = batch_signals_np[i, ch, :]
            # Compute CWT for the signal
            coefficients, _ = pywt.cwt(signal, frequencies, 'cmor1.5-1.0', sampling_period=1/sampling_frequency)
            coefficients = np.abs(coefficients)

            # Normalize the CWT coefficients
            if normalization == 'zscore':
                mean = np.mean(coefficients)
                std = np.std(coefficients)
                coefficients = (coefficients - mean) / std if std else coefficients
            elif normalization == 'minmax':
                min_val = np.min(coefficients)
                max_val = np.max(coefficients)
                coefficients = (coefficients - min_val) / (max_val - min_val) if max_val != min_val else coefficients

            cwt_channels.append(coefficients)

        cwt_channels_stacked = np.stack(cwt_channels, axis=0)  # Shape: (channels, frequencies, signal_length)
        cwt_batch.append(cwt_channels_stacked)

    cwt_batch_np = np.array(cwt_batch)  # Shape: (batch_size, channels, frequencies, signal_length)
    cwt_batch_np = np.transpose(cwt_batch_np, (0, 2, 3, 1))  # Transpose for shape: (batch_size, frequencies, signal_length, channels)
    cwt_batch_tensor = tf.convert_to_tensor(cwt_batch_np, dtype=tf.float32)
    return cwt_batch_tensor


# Select top 32 frequencies using Mutual Information + Random Forest
def select_best_frequencies(X_cwt, y, frequencies, top_k=50, final_k=32):
    """
    Select the top frequencies using Mutual Information and Random Forest feature importance.
    
    Args:
        X_cwt (np.array): CWT coefficients of shape (n_trials, frequencies, time_points, n_channels).
        y (np.array): Class labels for each trial.
        frequencies (np.array): Frequencies array corresponding to the input data.
        top_k (int): Number of top frequencies to pre-select with Mutual Information.
        final_k (int): Number of final frequencies to select after refinement.

    Returns:
        final_selected_indices (np.array): Indices of selected frequencies.
    """
    # Average across time and channels
    mean_cwt = np.mean(X_cwt, axis=(2, 3))  # Shape: (n_trials, frequencies)

    # Step 1: Mutual Information Preselection
    mi_scores = mutual_info_classif(mean_cwt, y)
    mi_indices = np.argsort(mi_scores)[-top_k:]  # Select top_k indices by MI scores

    # Step 2: Use Random Forest to refine the selection
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(mean_cwt[:, mi_indices], y)
    feature_importance = rf.feature_importances_

    # Select the most important `final_k` frequencies
    top_rf_indices = np.argsort(feature_importance)[-final_k:]  # Select the final_k most important features
    final_selected_indices = mi_indices[top_rf_indices]  # Map back indices to the original indices

    return final_selected_indices
