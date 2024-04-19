import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm

# Loading the dataset and displaying the first 10 rows
data = pd.read_csv("telecom_churn.csv")
data.head(10)



# Are there missing values?
data.isna().sum()



# Data types
data.dtypes



# Identifying the correlation between variables
data.corr()



# From the correlation matrix above, we notice that the variables CustServCalls, DayMins, and ContractRenewal are more relevant



# Checking how the data is distributed
num_true = len(data.loc[data['Churn'] == 0])
num_false = len(data.loc[data['Churn'] == 1])
print(f'Number of true cases: {num_true}, ({num_true / (num_true + num_false) * 100:.2f}%)')
print(f'Number of false cases (not churn): {num_false}, ({num_false / (num_true + num_false) * 100:.2f}%)')



# Splitting the data into training and testing sets
# Predictor variables
attributes = ['CustServCalls', 'DayMins', 'ContractRenewal']
# Target variable
target = ['Churn']
X = data[attributes].values
y = data[target].values



# Creating the objects
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)



# Displaying the results
print(f'{len(X_train) / len(data.index) * 100:.2f}% in the training data.')
print(f'{len(X_test) / len(data.index) * 100:.2f}% in the testing data.')



### Building the model with RANDOM FOREST

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train.ravel())

# Checking the training data
rf_train_predictions = random_forest.predict(X_train)
print(f'Training Accuracy: {accuracy_score(y_train, rf_train_predictions) * 100:.4f} %')

# Checking the accuracy on the testing data
rf_test_predictions = random_forest.predict(X_test)
print(f'Prediction Accuracy: {accuracy_score(y_test, rf_test_predictions) * 100:.4f} %')
