"""  stratify =Y is not needed in Y in linear regression in general 



 """


from matplotlib.pylab import LinAlgError
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 

insurance_dataset = pd.read_csv("Datasets/insurance.csv")

# print(insurance_dataset.isnull().sum()) # it doesn't have null values 

""" sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

 
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=insurance_dataset)
plt.title('Children Distribution')
plt.show()


plt.figure(figsize=(6,6))
sns.countplot(x='sex' , data=insurance_dataset)
plt.title('Sex Distribution')
plt.show() """

# print(insurance_dataset['sex'].value_counts())
""" sex
male      676
female    662
 """




# encoding the 

insurance_dataset.replace({'sex':{'male':0 , 'female':1}}, inplace = True )
insurance_dataset.replace({'smoker':{'yes':0 , 'no':1}}, inplace = True )
insurance_dataset.replace({'region':{'southeast':0 , 'southwest':1 ,'northeast':3 , 'northwest':4  }}, inplace = True )

X = insurance_dataset.drop(columns='charges' , axis = 1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train , Y_train)

training_data_prediciton = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediciton)
print("R squared value (Training) : ",r2_train)


testing_data_prediciton = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, testing_data_prediciton)
print("R squared value (Testing) : ",r2_test)


input_data = np.asarray((31,1,25.74,0,1,0)).reshape(1,-1)


prediction =  regressor.predict(input_data)
print("The insurance cost is ", prediction[0])

