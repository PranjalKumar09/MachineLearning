""" 
Work flow 

Car Data -> Data Pre Processing -> Train test split -> Linear & Lasso Regression 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import matplotlib.pyplot as plt

# loading data from csv file to pandas dataframee 

car_dataset = pd.read_csv('Datasets/car data.csv')

# print(car_dataset.shape) # (301, 9)


# print(car_dataset.info()) # this file has not null values


# print(car_dataset.Fuel_Type.value_counts()) # this file
""" 
Petrol    239
Diesel     60
CNG         2 """

# print(car_dataset.Seller_Type.value_counts()) # this file


car_dataset.replace({'Fuel_Type':{'Petrol': 0 , 'Diesel': 1, 'CNG': 2 } , 'Seller_Type' : {'Dealer' : 0, 'Individual' : 1} , 'Transmission' : {'Manual':0, 'Automatic':1}}, inplace = True )


X = car_dataset.drop(['Car_Name' , 'Selling_Price'] , axis = 1 ) # car name has no use
Y = car_dataset['Selling_Price'] 
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size= 0.1 , random_state= 2)



lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train , Y_train)




training_Data_prediction = lin_reg_model.predict(X_train)

# R squared error 
error_score = metrics.r2_score(Y_train , training_Data_prediction)
print("R squared error (train data): " , error_score )




# visualize actual price and predicated price 

plt.scatter(Y_train, training_Data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Train     Actual vs Predicated Graph")
plt.show()

# predication on training data



training_Data_prediction = lin_reg_model.predict(X_test)

# R squared error 
error_score = metrics.r2_score(Y_test , training_Data_prediction)
print("R squared error (test data): " , error_score )

plt.scatter(Y_test, training_Data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Test    Actual vs Predicated Graph")
plt.show()

plt.title("Actual Prices vs Predicated Prices")


