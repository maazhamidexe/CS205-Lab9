# CS205-Lab9
# Telecom Churn Prediction Model
This repository contains a Python-based model for predicting customer churn in the telecom industry using a dataset named telecom_churn.csv. The model utilizes machine learning techniques to analyze customer behavior and predict whether a customer is likely to churn or not.

# Dataset Overview
The dataset telecom_churn.csv contains information about telecom customers, including their service usage, contract renewal, and whether they churned or not. The model focuses on three key predictors: CustServCalls, DayMins, and ContractRenewal.

# Model Architecture
The model is built using the Random Forest Classifier from the sklearn.ensemble module. This classifier is chosen for its ability to handle a large number of features and its robustness against overfitting.

# Key Features
Data Exploration: The model begins by exploring the dataset, checking for missing values, examining data types, and identifying correlations between variables.
Data Preprocessing: The dataset is preprocessed to ensure it is ready for model training. This includes handling missing values and selecting relevant features for prediction.
Model Training: The model is trained using a 60-40 split of the dataset, with 60% of the data used for training and 40% for testing.
Model Evaluation: The model's performance is evaluated using accuracy score, confusion matrix, and classification report to understand its effectiveness in predicting customer churn.

# Installation
To run this model, you need to have Python installed on your system. Additionally, you will need to install the following Python libraries:

pandas
numpy
scikit-learn
You can install these libraries using pip:

pip install pandas numpy scikit-learn
# Usage
Clone this repository to your local machine.
Ensure you have the telecom_churn.csv dataset in the same directory as the Python script.
Run the Python script to train the model and evaluate its performance.
# Results
The accuracy of the model is 83%
