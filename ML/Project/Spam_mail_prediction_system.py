""" 
Work Flow -> Mail Data -> Data pre processing -> Train Test Split -> Logistic Regression Model

"""
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression





raw_data = pd.read_csv("Datasets/mail_data.csv")


# replace null values with null string
mail_data = raw_data.where((pd.notnull(raw_data)) , '')

# print (mail_data.head())



# label encoding 
mail_data.loc[mail_data["Category"] == "spam" , "Category" ,] = 0 
mail_data.loc[mail_data["Category"] == "ham" , "Category" ,] = 1

X = mail_data.loc[: , "Message"]
Y = mail_data.loc[: , "Category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# print (X)
# print (Y)





# transform text data to feature vectors that can be used as input to the logistic regression  
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True) # fitting all data 

X_train_feautres = feature_extraction.fit_transform(X_train) # transforming the training data
X_test_feautres = feature_extraction.transform(X_test) # transforming the test data

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# print (X_train_feautres)


model = LogisticRegression()
model.fit(X_train_feautres, Y_train)

X_train_prediction = model.predict(X_train_feautres)
training_data_accuracy = accuracy_score(Y_train , X_train_prediction)
print("Accuracy of training data : " , training_data_accuracy)

X_test_prediction = model.predict(X_test_feautres)
test_data_accuracy = accuracy_score(Y_test , X_test_prediction)
print("Accuracy of test data : " , test_data_accuracy)

# Building Predicating System
input_mail = feature_extraction.transform(["Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's left over dinner ? Do you feel my Love ?"])
# covert 

predication =  model.predict(input_mail)
if predication[0] == 1 : print ("Ham mail")
else : print ("Spam mail")