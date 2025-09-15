"""
This script is for advanced model evaluation and hyperparameter tuning.

It performs the following steps:
1. Loads the training data from 'train.csv'.
2. Splits the data into training and testing sets.
3. Creates and trains a scikit-learn pipeline (TfidfVectorizer and MultinomialNB).
4. Evaluates the model using accuracy, a classification report, and a confusion matrix.
5. Performs hyperparameter tuning using GridSearchCV to find the best 'alpha' for the MultinomialNB classifier.
6. Prints the best parameters and cross-validation score found by GridSearchCV.

This script is intended for experimentation and analysis, not for production use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(filepath):
    """Loads data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    return pd.read_csv(filepath)

def train_and_evaluate(data):
    """Trains and evaluates a document classification model.

    Args:
        data (pandas.DataFrame): The training data, containing 'text' and 'category' columns.

    Returns:
        A trained scikit-learn pipeline model.
    """
    X = data['text']
    y = data['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

def perform_grid_search(model, data):
    """Performs a grid search to find the best hyperparameters.

    Args:
        model: The scikit-learn model to tune.
        data (pandas.DataFrame): The training data.
    """
    X = data['text']
    y = data['category']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'multinomialnb__alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("\nBest parameters from GridSearchCV:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

def main():
    """Main function to run the model evaluation and tuning process."""
    data = load_data('train.csv')
    model = train_and_evaluate(data)
    perform_grid_search(model, data)

if __name__ == '__main__':
    main()
