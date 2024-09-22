from eeg.loader import EEGDataLoader
from eeg.preprocessor import EEGPreprocessor
from eeg.feature_extractor import EEGFeatureExtractor
from eeg.classifier import EEGClassifier

from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info(f"Starting EEG Classification Pipeline")

    # Configuration
    data_dir = 'data'
    column_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5',
                    'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
    fs = 256  # Sampling frequency in Hz
    lowcut = 0.5
    highcut = 30

    # Data Loading
    data_loader = EEGDataLoader(data_dir, column_names)
    eeg_data = data_loader.load_data()
    print(eeg_data.head())

    # Preprocessing
    preprocessor = EEGPreprocessor(fs)
    eeg_data = preprocessor.apply_filter(eeg_data, 'Fp1', lowcut, highcut)

    # Feature Extraction
    feature_extractor = EEGFeatureExtractor(fs)
    features_df = feature_extractor.extract_features_from_data(eeg_data)
    print(features_df.head())

    # Assuming 'target_label' is the column name for the label you want to classify
    y = eeg_data['target_label']  # Replace 'target_label' with your actual target column
    X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.3, random_state=42)

    # Model Training and Evaluation
    classifier = EEGClassifier()
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    classifier.evaluate(y_test, y_pred)

    logging.info(f"EEG Classification Pipeline Completed")


if __name__ == "__main__":
    main()
