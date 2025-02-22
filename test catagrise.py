import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the data
data = pd.read_csv('train.csv')

# Step 2: Split the data into features and labels
X = data['text']  # Features (document text)
y = data['category']  # Labels (categories)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a pipeline that combines TfidfVectorizer and MultinomialNB
model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Hyperparameter tuning using GridSearchCV
param_grid = {
    'multinomialnb__alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
    # Example hyperparameters for Naive Bayes
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest parameters from GridSearchCV:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
