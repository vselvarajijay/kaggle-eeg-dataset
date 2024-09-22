from scipy.signal import butter, lfilter

class EEGPreprocessor:
    def __init__(self, fs):
        self.fs = fs

    def butter_bandpass(self, lowcut, highcut, order=5):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        y = lfilter(b, a, data)
        return y

    def apply_filter(self, eeg_data, column, lowcut, highcut):
        eeg_data[f'{column}_filtered'] = self.bandpass_filter(eeg_data[column], lowcut, highcut)
        return eeg_data
