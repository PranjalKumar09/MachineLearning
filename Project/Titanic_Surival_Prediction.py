import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_Data = pd.read_csv('Datasets/csv/train.csv')

# print(titanic_Data.shape) # (891, 12)

# print(titanic_Data.isnull().sum()) 
""" 
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64 """

# handling the missing values 

titanic_Data = titanic_Data.drop(columns = "Cabin", axis = 1)
titanic_Data['Age'].fillna(titanic_Data['Age'].mean()  , inplace=True)
titanic_Data['Embarked'].fillna(titanic_Data['Embarked'].mode()[0], inplace=True) 
# print(titanic_Data.isnull().sum()) # now no null values

titanic_Data.replace({'Sex' : {'male' : 0 , 'female' :1} , 'Embarked' : {'S':0 , 'C': 1 , 'Q': 2}} , inplace= True)

X = titanic_Data.drop(columns = ['PassengerId' , 'Name' , 'Ticket', 'Survived'],  axis = 1) 
Y = titanic_Data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)

training_Data_accuracy =  accuracy_score(Y_train,X_train_prediction )
print("Accuracy score of train data : ", training_Data_accuracy)

X_test_prediction = model.predict(X_test)

testing_Data_accuracy =  accuracy_score(Y_test,X_test_prediction )
print("Accuracy score of train data : ", testing_Data_accuracy)