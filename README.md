# Codsoft
 Codsoft 2024 July 
# SMS Spam Classifier

This project is an SMS spam classifier using a logistic regression model. The project involves loading and cleaning a dataset, preprocessing the text data, training a logistic regression model, evaluating its performance, and saving/loading the model for future use.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Usage](#usage)
- [Contributing](#contributing)


## Installation

To run this project, you'll need to have Python installed along with the following libraries:

- pandas
- numpy
- re
- string
- nltk
- scikit-learn
- matplotlib
- seaborn
- joblib

You can install these packages using pip:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib
```
## Dataset
The dataset used in this project is a collection of SMS messages labeled as "spam" or "ham" (not spam). The dataset is loaded from a CSV file with the following structure:

label: The label indicating whether the message is spam or ham.
text: The SMS message text.

Link to [Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
## Data Preprocessing
### Cleaning the Data
The dataset is cleaned by removing unnecessary columns and renaming the remaining columns for clarity.

### Text Preprocessing
The text data is preprocessed by:

- Converting text to lowercase
- Removing text in square brackets
- Removing links
- Removing punctuation
- Removing words containing numbers
- Removing stopwords
- Stemming words
##  Model Training
 
The preprocessed text data is split into training and testing sets. The training data is then transformed using CountVectorizer and TfidfTransformer.

A logistic regression model is trained using Grid Search Cross-Validation to find the best hyperparameters.

## Model Evaluation
The model's performance is evaluated using accuracy, classification report, and confusion matrix.

## Saving and Loading the Model
The trained model and vectorizers are saved using joblib for future use.

## Usage
```bash
# Save the best model and vectorizer
joblib.dump(best_model, 'sms_spam_classifier.pkl')
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')
joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')
```
## Load the Model and Vectorizer

```bash
import joblib

# Load the model and vectorizer
loaded_model = joblib.load('sms_spam_classifier.pkl')
loaded_vectorizer = joblib.load('count_vectorizer.pkl')
loaded_tfidf_transformer = joblib.load('tfidf_transformer.pkl')

# Function to classify new messages
def classify_message(message):
    cleaned_message = preprocess_text(message)
    message_count = loaded_vectorizer.transform([cleaned_message])
    message_tfidf = loaded_tfidf_transformer.transform(message_count)
    prediction = loaded_model.predict(message_tfidf)
    return prediction[0]

```

## Example Usage

```bash
# Example usage
message = "Congratulations! You've won a $1000 gift card. Call now!"
print(f'New Message: {message}')
print(f'Classification: {classify_message(message)}')
```
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
