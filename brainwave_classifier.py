import pandas as pd
import numpy as np
import os


# Load all CSV files
data_dir = 'data'
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]

# Combine all files into one DataFrame
data_frames = [pd.read_csv(file) for file in file_list]
eeg_data = pd.concat(data_frames, ignore_index=True)

# Display the first few rows of the dataset
print(eeg_data.head())
