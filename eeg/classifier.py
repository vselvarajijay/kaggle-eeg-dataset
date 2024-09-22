from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EEGClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, class_weight='balanced')


    def train(self, X_train, y_train):
        logging.info("Training the classifier")
        self.model.fit(X_train, y_train)
        logging.info("Training completed")

    def predict(self, X_test):
        logging.info("Predicting labels")
        prediction = self.model.predict(X_test)
        logging.info("Prediction completed")
        return prediction

    def evaluate(self, y_test, y_pred):
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

