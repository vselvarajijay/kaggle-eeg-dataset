import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
import pywt

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EEGFeatureExtractor:
    def __init__(self, fs):
        self.fs = fs

    def extract_features(self, eeg_signal):
        eeg_signal = np.array(eeg_signal)

        if eeg_signal.ndim != 1:
            raise ValueError("eeg_signal should be a 1-dimensional array.")

        features = {}

        # Power Spectral Density (PSD)
        freqs = np.fft.fftfreq(len(eeg_signal), d=1 / self.fs)
        psd = np.abs(np.fft.fft(eeg_signal))**2

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

    def extract_features_from_data(self, eeg_data):
        logging.info("Starting feature extraction from EEG data")
        feature_list = []
        total_rows = len(eeg_data)

        # Initialize a variable to track the last logged percentage
        last_logged_percent = 0

        for i in range(total_rows):
            signal = eeg_data.iloc[i, :].values
            features = self.extract_features(signal)
            feature_list.append(features)

            # Calculate the current percentage completed
            percent_complete = ((i + 1) / total_rows) * 100

            # Log the progress for every 1% increase
            if percent_complete >= last_logged_percent + 1 or i == total_rows - 1:
                last_logged_percent = int(percent_complete)
                logging.info(f"Feature extraction progress: {percent_complete:.2f}% complete")

        logging.info("Feature extraction completed")
        return pd.DataFrame(feature_list)
