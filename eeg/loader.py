import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EEGDataLoader:
    def __init__(self, data_dir, column_names):
        self.data_dir = data_dir
        self.column_names = column_names

    def load_data(self):
        logging.info(f"Loading EEG data from {self.data_dir}")

        file_list = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.csv')]
        data_frames = [pd.read_csv(file, header=None, names=self.column_names) for file in file_list]
        eeg_data = pd.concat(data_frames, ignore_index=True)

        logging.info(f"EEG data loaded successfully. Shape: {eeg_data.shape}")
        return eeg_data
