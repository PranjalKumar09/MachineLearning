"""
By Machine Vector Machine 
It is supervised learning algorithm

Work Flow ->

Diabetes Data -> Data Pre Processing Model -> Train Test Split -> Support Vecotor Machine Classifier

It will tell whether perosn is diabitics or not

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection and analysis 
# PIMA Diabitics Dataset



# loading the dataset dataset to a pandas dataframe

diabetes_dataset = pd.read_csv('Datasets/diabetes.csv')


# print(diabetes_dataset.head())

# print(diabetes_dataset.shape) # (768 , 9)


# getting the statiscal measure of the data

# print(diabetes_dataset.describe())


# print(diabetes_dataset["Outcome"].value_counts())
# Outcome
# 0    500
# 1    268

# 0 -> Non - Diabeitc
# 1 -> Diabeitc

# print(diabetes_dataset.groupby("Outcome").mean())


# seperating the data and labels 

X  = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y  = diabetes_dataset[ 'Outcome']


# print(X)



# Data Standardization


scalar = StandardScaler()
scalar.fit(X)  # It is must to do it


Standartized_data = scalar.transform(X)

# print (X)
# print (Y)


X_Train , X_Test , Y_Train , Y_Test = train_test_split(X , Y , test_size= 0.2, stratify= Y , random_state=2 ) # startify is to ignore case in which  only one thing going 


print(X.shape , X_Test.shape , X_Train.shape) # (768, 8) (154, 8) (614, 8)





# Training the Model 
classifer = svm.SVC(kernel= 'linear')

# training the support vector Machine Classifier

classifer.fit(X_Train, Y_Train)




# Model Evaluation

# Accuracy Score

X_train_prediction = classifer.predict(X_Train)
training_data_predicition = accuracy_score(X_train_prediction , Y_Train)

print("Accuracy Score of the training data : " , training_data_predicition)


X_test_prediction = classifer.predict(X_Test)
test_data_predicition = accuracy_score(X_test_prediction , Y_Test)

print("Accuracy Score of the test data : " , test_data_predicition)




input_data = (1,97,66,15,140,23.2,0.487,22)

# changing data as input array to numppy array 
input_data_as_numpy_array = np.asarray(input_data)


# reshahpe the array as we can preicating of one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


#standaritze hte input data 
std_data = scalar.transform(input_data_reshaped)

print(std_data)

prediction = classifer.predict(std_data)

print("The Person is Diabiatic ") if prediction[0] == 1 else print("The Person is not Diabiatic ")