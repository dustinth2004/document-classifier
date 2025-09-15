"""
This script trains a document classification model.

It performs the following steps:
1. Loads the training data from 'train.csv'.
2. Splits the data into training and testing sets.
3. Creates a scikit-learn pipeline that combines a TfidfVectorizer and a MultinomialNB classifier.
4. Trains the model on the training data.
5. Saves the trained model to 'trained_model.pkl'.
6. Evaluates the model on the test data and prints a classification report.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.metrics import classification_report

def load_data(filepath):
    """Loads data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    return pd.read_csv(filepath)

def train_model(data):
    """Trains a document classification model.

    Args:
        data (pandas.DataFrame): The training data, containing 'text' and 'category' columns.

    Returns:
        A trained scikit-learn pipeline model.
    """
    X = data['text']
    y = data['category']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, data):
    """Evaluates the performance of the model.

    Args:
        model: The trained scikit-learn model.
        data (pandas.DataFrame): The test data, containing 'text' and 'category' columns.
    """
    X = data['text']
    y = data['category']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
    """Main function to run the model training and evaluation process."""
    data = load_data('train.csv')
    model = train_model(data)
    joblib.dump(model, 'trained_model.pkl')
    evaluate_model(model, data)

if __name__ == '__main__':
    main()
