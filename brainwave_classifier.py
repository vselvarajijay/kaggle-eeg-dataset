import pandas as pd
import numpy as np
import os

from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis
import pywt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


"""
Notes
Replace path_to_your_dataset with the path to your EEG dataset.
Modify target_label to the column representing your classification labels.
Adjust feature extraction and model parameters based on your dataset's complexity.
"""


# Load all CSV files
data_dir = 'data'
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]

# Combine all files into one DataFrame
data_frames = [pd.read_csv(file) for file in file_list]
eeg_data = pd.concat(data_frames, ignore_index=True)

# Display the first few rows of the dataset
print(eeg_data.head())


# Define the bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Example of filtering data for one channel
fs = 256  # Sampling frequency in Hz
eeg_data['Fp1_filtered'] = bandpass_filter(eeg_data['Fp1'], 0.5, 30, fs)



# Function to extract features
def extract_features(eeg_signal):
    features = {}
    
    # Power Spectral Density (PSD)
    freqs, psd = np.abs(np.fft.fft(eeg_signal)), np.fft.fft(eeg_signal)
    features['mean_psd'] = np.mean(psd)
    features['std_psd'] = np.std(psd)
    
    # Wavelet Transform (using 'db4' wavelet)
    coeffs = pywt.wavedec(eeg_signal, 'db4', level=5)
    features['mean_wavelet'] = np.mean(coeffs[0])
    features['std_wavelet'] = np.std(coeffs[0])
    
    # Statistical features
    features['mean'] = np.mean(eeg_signal)
    features['std'] = np.std(eeg_signal)
    features['skew'] = skew(eeg_signal)
    features['kurtosis'] = kurtosis(eeg_signal)
    
    return features

# Extract features for each sample (example for the Fp1 channel)
feature_list = []
for i in range(len(eeg_data)):
    signal = eeg_data.loc[i, 'Fp1_filtered']
    features = extract_features(signal)
    feature_list.append(features)

features_df = pd.DataFrame(feature_list)
print(features_df.head())






# Assuming 'target_label' is the column name for the label you want to classify
X = features_df
y = eeg_data['target_label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))





# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

