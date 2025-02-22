import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib  # Correct import for joblib
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv('train.csv')  # Make sure to replace 'your_file.csv' with your actual file path

# Split the data into features and labels
X = data['text']  # Features (document text)
y = data['category']  # Labels (categories)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that combines TfidfVectorizer and MultinomialNB
model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'trained_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
