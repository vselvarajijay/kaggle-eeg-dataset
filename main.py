from eeg.loader import EEGDataLoader
from eeg.preprocessor import EEGPreprocessor
from eeg.feature_extractor import EEGFeatureExtractor
from eeg.classifier import EEGClassifier

from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def determine_state(row):
    # Extract frequency domain power features
    delta_power = row['delta_power']
    theta_power = row['theta_power']
    alpha_power = row['alpha_power']
    beta_power = row['beta_power']
    gamma_power = row['gamma_power']

    # Simple thresholds based on power band dominance
    if alpha_power > beta_power and alpha_power > theta_power and alpha_power > gamma_power:
        return 0  # Calm
    elif beta_power > alpha_power and beta_power > theta_power and beta_power > gamma_power:
        return 1  # Focused
    elif theta_power > alpha_power and theta_power > beta_power and theta_power > gamma_power:
        return 2  # Stressed
    elif gamma_power > alpha_power and gamma_power > beta_power and gamma_power > theta_power:
        return 2  # Stressed (Gamma-dominant stress)

    # Default to a focused state if there's no clear dominance
    return 1  # Focused


def main():
    logging.info("Starting EEG Classification Pipeline")

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

    # Split the data first
    X_train, X_test = train_test_split(eeg_data, test_size=0.3, random_state=42)

    # Preprocessing
    preprocessor = EEGPreprocessor(fs)

    # Apply filtering to train and test sets
    for column in column_names:
        X_train[column + '_filtered'] = preprocessor.apply_filter(X_train, column, lowcut, highcut)
        X_test[column + '_filtered'] = preprocessor.apply_filter(X_test, column, lowcut, highcut)

    # Feature Extraction
    feature_extractor = EEGFeatureExtractor(fs)
    train_features = feature_extractor.extract_features_from_data(X_train[[col + '_filtered' for col in column_names]])
    test_features = feature_extractor.extract_features_from_data(X_test[[col + '_filtered' for col in column_names]])

    # Create Target Labels using the training set features
    train_features['target_label'] = train_features.apply(determine_state, axis=1)
    test_features['target_label'] = test_features.apply(determine_state, axis=1)

    # Extract target labels
    y_train = train_features['target_label']
    y_test = test_features['target_label']

    # Remove the target label from the feature sets
    X_train = train_features.drop(columns=['target_label'])
    X_test = test_features.drop(columns=['target_label'])

    # Check for data leakage
    overlap = pd.merge(X_train, X_test, how='inner')
    print(f">>> Number of overlapping rows between training and test sets: {len(overlap)}")

    # Verify that Target Labels are Meaningfully Different
    print(">>> Training label distribution:\n", y_train.value_counts())
    print(">>> Test label distribution:\n", y_test.value_counts())

    # Model Training and Evaluation
    classifier = EEGClassifier()
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    classifier.evaluate(y_test, y_pred)

    logging.info("EEG Classification Pipeline Completed")

if __name__ == "__main__":
    main()
