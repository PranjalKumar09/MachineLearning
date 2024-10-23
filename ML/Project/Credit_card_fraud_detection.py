import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

crdit_card_data = pd.read_csv("Datasets/credit_data.csv")

# print(crdit_card_data.tail())

# print(crdit_card_data.isnull().sum()) # no null values 


X = crdit_card_data.drop(columns='Class' , axis   = 1)
Y = crdit_card_data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.1, stratify=Y)


model = LogisticRegression()

model.fit(X_train, Y_train)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print ("Accuracy on test data ", testing_data_accuracy)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
print ("Accuracy on training data ", training_data_accuracy)

