from scipy.signal import butter, lfilter

class EEGPreprocessor:
    def __init__(self, fs):
        self.fs = fs

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_filter(self, eeg_data, column, lowcut, highcut, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, self.fs, order=order)

        # Apply filter to each row for the specified column
        filtered_column = lfilter(b, a, eeg_data[column].values)

        return filtered_column