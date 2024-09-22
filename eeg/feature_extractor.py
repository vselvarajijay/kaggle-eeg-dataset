import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis

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


        """
        # 1. Time-Domain Features
        features['mean'] = np.mean(eeg_signal)
        features['std'] = np.std(eeg_signal)
        features['skew'] = skew(eeg_signal)
        features['kurtosis'] = kurtosis(eeg_signal)
        features['rms'] = np.sqrt(np.mean(eeg_signal**2))
        """

        # 2. Frequency-Domain Features
        freqs = np.fft.fftfreq(len(eeg_signal), d=1 / self.fs)
        psd = np.abs(np.fft.fft(eeg_signal))**2


        # Only consider positive frequencies
        positive_freqs = freqs > 0
        freqs = freqs[positive_freqs]
        psd = psd[positive_freqs]


        # Extract band power for Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), and Gamma (>30 Hz)
        features['delta_power'] = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        features['theta_power'] = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        features['alpha_power'] = np.mean(psd[(freqs >= 8) & (freqs < 12)])
        features['beta_power'] = np.mean(psd[(freqs >= 12) & (freqs < 30)])
        features['gamma_power'] = np.mean(psd[(freqs >= 30)])

        # 3. Dominant Frequency
        dominant_freq_index = np.argmax(psd)
        features['dominant_frequency'] = freqs[dominant_freq_index]

        # 4. Time-Frequency Domain Features (Wavelet Transform using 'db4' wavelet)
        """
        coeffs = pywt.wavedec(eeg_signal, 'db4', level=5)
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_mean_{i}'] = np.mean(coeff)
            features[f'wavelet_std_{i}'] = np.std(coeff)
            features[f'wavelet_skew_{i}'] = skew(coeff)
            features[f'wavelet_kurtosis_{i}'] = kurtosis(coeff)
        """

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