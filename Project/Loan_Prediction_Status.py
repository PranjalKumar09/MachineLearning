""" 
Data -> Data pre processing -> Train Test split -> Support vector machine model 

"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score





# Data Collection and Data Processing
loan_dataset = pd.read_csv("Datasets/loan_status_dataset.csv")



# dropping the missing values
loan_dataset = loan_dataset.dropna()


loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_dataset.replace(to_replace='3+', inplace=True, value=4)



# Convert categorical columns to numerical values
loan_dataset.replace({"Married": {'No': 0, 'Yes': 1},  
    "Gender": {'Male': 1, 'Female': 0},
    "Self_Employed": {'No': 0, 'Yes': 1},
    "Property_Area": {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    "Education": {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

# Now we don't need the 'Loan_ID' column
# Separating the data and label
X = loan_dataset.drop(columns=['Loan_Status', 'Loan_ID'], axis=1)
Y = loan_dataset['Loan_Status']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.1, stratify=Y)

# Training the model - Support Vector Machine model

classifier  = svm.SVC(kernel= 'linear')

# print("Missing values in X_train:\n", X_train.isnull().sum())

# a = Y_train.values.ravel()
classifier.fit(X_train, Y_train)





X_test_prediction = classifier.predict(X_test)
training_data_acccuracy = accuracy_score(X_test_prediction  ,Y_test)
print ("Accuracy on test data " , training_data_acccuracy)

X_train_prediction = classifier.predict(X_train)
training_data_acccuracy = accuracy_score(X_train_prediction  ,Y_train)
print ("Accuracy on training data " , training_data_acccuracy)