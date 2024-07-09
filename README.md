# Codsoft
 Codsoft 2024 July 

Certainly! Here's a README.md template for the IMDb genre classification project:

# IMDb Genre Classification Project

## Project Overview
This project focuses on building a machine learning model to classify movie genres based on their descriptions using the IMDb dataset. The goal is to predict genres for movies using natural language processing techniques.

## Dataset
The dataset used in this project consists of two main files:

train_data.txt: Training dataset containing movie IDs, titles, genres, and descriptions.
test_data.txt: Test dataset containing movie IDs, titles, and descriptions.
Link to [Dataset](https://www.kaggle.com/code/hamzasafwan/movie-genre-classification)
## Models Used
The model employed in this project utilizes a combination of text preprocessing, TF-IDF vectorization, and Logistic Regression for classification.

## Installation
To run the code for this project, ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn
```
## Usage
Training the Model
Load and Preprocess Data: The training data (train_data.txt) is loaded and preprocessed to extract descriptions and genres.

TF-IDF Vectorization: Text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to convert descriptions into numerical features.

Model Training: A Logistic Regression model is trained on the TF-IDF transformed data.

## Example Code
```bash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load the training data
train_data = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

# Preprocess the data
X_train = train_data['DESCRIPTION']
y_train = train_data['GENRE']

# Create a pipeline with TF-IDF vectorizer and Logistic Regression classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model using joblib
import joblib
model_filename = 'imdb_genre_classifier.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved as {model_filename}")
```
## Predicting with the Model
Load the Model: Load the saved model (imdb_genre_classifier.pkl).

Predict on Test Data: Use the trained model to predict genres for movies in the test dataset (test_data.txt).

Save Predictions: Save the predicted genres to test_predictions.csv.

## Example Code
```bash
# Load the test data
test_data = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['ID', 'TITLE', 'DESCRIPTION'])

# Load the trained model
loaded_model = joblib.load('imdb_genre_classifier.pkl')

# Make predictions on the test data
predictions = loaded_model.predict(test_data['DESCRIPTION'])

# Prepare the output DataFrame
output = pd.DataFrame({'ID': test_data['ID'], 'TITLE': test_data['TITLE'], 'PREDICTED_GENRE': predictions})

# Save the predictions to a CSV file
output.to_csv('test_predictions.csv', index=False)
print("Predictions saved to test_predictions.csv")
```
## Results
The model achieves classification results which can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.
