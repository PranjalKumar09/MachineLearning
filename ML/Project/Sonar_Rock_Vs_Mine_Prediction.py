""" 
Work Flow -> 
Collect Sonar Data -> Data Pre Processing -> Text Test Split -> Logistic Regression Model 

New Data -> Trained Logistic Regression Model -> Rock or Mine


"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Data Processing


# loading the dataset  to pandas dataframe 

sonar_data =  pd.read_csv("Datasets/Copy of sonar data.csv" , header = None)

# print(sonar_data.shape) # (208 , 61)
# print ( sonar_data.describe() ) # describe --> statictal of deta

# Note in data set 60th  column is result where written as R of B

# print (sonar_data.groupby(60).mean())



# seprating data and labels 
X = sonar_data.drop(axis =  1,  columns = 60)
Y = sonar_data[60]



# Testing and Test data

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.1 , stratify= Y, random_state = 1)



# model traning -> Logical Regression

model = LogisticRegression()


#creating the logistic regression model with training data

model.fit(X_train, Y_train)




# Model Evaluation 
# accuracy on training data 


X_train_prediction = model.predict(X_train)

training_data_prediction = accuracy_score(X_train_prediction , Y_train)

print ( "Accuracy on training data : " , training_data_prediction)

X_text_prediction = model.predict(X_test)
training_data_prediction = accuracy_score(X_text_prediction , Y_test)

print ( "Accuracy on test data : " , training_data_prediction)


input_data = (0.0264,0.0071,0.0342,0.0793,0.1043,0.0783,0.1417,0.1176,0.0453,0.0945,0.1132,0.0840,0.0717,0.1968,0.2633,0.4191,0.5050,0.6711,0.7922,0.8381,0.8759,0.9422,1.0000,0.9931,0.9575,0.8647,0.7215,0.5801,0.4964,0.4886,0.4079,0.2443,0.1768,0.2472,0.3518,0.3762,0.2909,0.2311,0.3168,0.3554,0.3741,0.4443,0.3261,0.1963,0.0864,0.1688,0.1991,0.1217,0.0628,0.0323,0.0253,0.0214,0.0262,0.0177,0.0037,0.0068,0.0121,0.0077,0.0078,0.0066)

#changing the  input data to numpy array array

input_data = np.array(input_data)

input_data = input_data.reshape(1, -1) # we are predicating for one instance


prediction = model.predict(input_data)


print("The object is a rock") if prediction[0] == "R" else print("The object is mine")