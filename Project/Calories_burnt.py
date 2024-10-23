import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from xgboost import XGBRegressor
from  sklearn.model_selection import train_test_split

calories = pd.read_csv('/content/calories.csv')
exercise_data = pd.read_csv('/content/exercise.csv')

calories_data = pd.concat([exercise_data , calories['Calories']] , axis = 1) 
# there is no null data 

calories_data.replace({'Gender' : {'male' : 0,  'female' : 1}} , inplace = True)


X = calories_data.drop(columns= ['User_ID' , 'Calories'] , axis = 1)
Y = calories_data['Calories']

X_train  , X_test  , Y_train , Y_test = train_test_split(X, Y  , test_size=0.2 , random_state= 2)
model = XGBRegressor()
model.fit(X_train ,Y_train)

training_data_prediction = model.predict(X_train)
r2_train = metrics.r2_score(Y_train , training_data_prediction)
print('R squared value (train) = ',r2_train)


testing_data_prediction = model.predict(X_test)
r2_test = metrics.r2_score(Y_test , testing_data_prediction)
print('R squared value (test) = ',r2_test)
