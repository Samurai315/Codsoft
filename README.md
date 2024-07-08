# Codsoft
 Codsoft 2024 July 
# Customer Churn Prediction

This project focuses on predicting customer churn using various machine learning models. The primary goal is to identify customers who are likely to leave the bank, helping the bank to take preventive measures.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Further Training](#further-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Customer churn prediction is a critical aspect for businesses, especially in the banking sector. This project implements multiple machine learning models to predict customer churn, including Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking Classifier. The models are trained on the `Churn_Modelling.csv` dataset and evaluated for accuracy and classification performance.

## Dataset

The dataset used in this project is `Churn_Modelling.csv`, which contains the following columns:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target variable)

The test dataset is `test.csv`.

## Models Used

The following models were used in this project:
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Stacking Classifier (combination of Random Forest, Gradient Boosting, XGBoost with Logistic Regression as the final estimator)
- Sequential CNN Model
## Installation

To install the necessary libraries, run:

```bash
pip install pandas numpy scikit-learn xgboost joblib
```
## Usage
Train the models: The models are trained using the provided Churn_Modelling.csv dataset. The best performing model is saved using joblib.

Make predictions on the test set: Load the saved model and preprocess the test data. Make predictions and save the results to test_predictions.csv.

## Further Training
To further train the saved model on more external data, load the model using joblib and continue training on the new dataset.

```bash
import joblib

# Load the saved model
loaded_stacking_pipeline = joblib.load('stacking_pipeline_model.pkl')

# Train on new data
X_new, y_new = ...  # Load your new dataset
loaded_stacking_pipeline.fit(X_new, y_new)
```
## Results
The best model, Stacking Classifier, achieved the following performance on the validation set:

Accuracy: 0.8720
```bash
Classification Report:
            precision    recall  f1-score   support

         0       0.89      0.96      0.92      1607
         1       0.77      0.50      0.60       393

  accuracy                           0.87      2000
 macro avg       0.83      0.73      0.76      2000
```

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
