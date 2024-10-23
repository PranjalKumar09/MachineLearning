""" 

search  heart ecg _Wave in google


Gold Price -> Data Pre Processing -> Data Analysis -> Test Train Split -> Random forest Regressor



"""
from random import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_dataset = pd.read_csv("Datasets/heart_diseas.csv")

# no null values in heart_dataset

# print(heart_dataset.describe())

# checkig the distribution of Target values



# print(heart_dataset['target'].value_counts())
""" target
1    165
0    138 """



X = heart_dataset.drop(columns='target')
Y = heart_dataset['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify = Y )


model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
# print ( "Accuracy on training data : ", training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)
# print ( "Accuracy on test data : ", test_data_accuracy)





input_data = np.asarray((42,1,0,140,226,0,1,178,0,0,2,0,2)).reshape(1,-1)
prediction =  model.predict(input_data)
if prediction[0] == 0 :
    print("The person does not have Heart Disease")
else: 
    print("The person  have Heart Disease")
    