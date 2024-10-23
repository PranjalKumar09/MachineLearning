"""
Medical knowledge (Not necessary)

    Benign tumour (cant be move to other part of body , not dangerous as malignant  )
        * Non cancerous

    Malignant tumour (can move to other part of body , dangerous)
        * Cancerous
        
    
WorkFlow
    Dataset -> Data Pre Processing -> Train Text Split -> Logisitic Regression Model
    
    New Data -> Trained Logisitc Regression Model -> Bengin Or Malignant
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# loading data from sklearn

dataset = pd.read_csv("Datasets/csv/data.csv")


# print(dataset.head())
# print(dataset.info())  # dont have missing values
# print(dataset.isnull().sum()) # dont have missing values
# print(dataset.describe()) 
dataset['diagnosis'].replace({'B':1, 'M':0} , inplace=True)




X = dataset.drop(columns=['diagnosis' , 'id', 'Unnamed: 32']  , axis = 1)

Y = dataset['diagnosis']

# print(X)
# print(Y)


X_train , X_test , Y_train , Y_test = train_test_split(X , Y, test_size = 0.2 , random_state=2)

# print(X_train.shape  , X_test.shape  , X.shape) # (455, 30) (114, 30) (569, 30)
model   = LogisticRegression()
model.fit(X_train, Y_train)
# accuracy score  

X_train_prediction = model.predict(X_train)
training_data_score = accuracy_score(Y_train, X_train_prediction)

print("Accuracy score for training data: ", training_data_score)

X_test_prediction = model.predict(X_test)
testing_data_score = accuracy_score(Y_test, X_test_prediction)
print("Accuracy score for testing data : " , testing_data_score)



input_data = np.asarray((13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)).reshape(1,-1)
print("Breast-Cancer is Malignant ") if model.predict(input_data)[0]==0 else print("Breast-Cancer is Belign")

