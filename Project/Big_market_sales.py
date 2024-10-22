import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from xgboost import XGBRegressor
from  sklearn.model_selection import train_test_split



big_mart_data = pd.read_csv('/content/Train.csv')
# big_mart_data.shape # ((8523, 12))
# print(big_mart_data.isnull().sum())
''' 

Item_Identifier                 0
Item_Weight                  1463
Item_Fat_Content                0
Item_Visibility                 0
Item_Type                       0
Item_MRP                        0
Outlet_Identifier               0
Outlet_Establishment_Year       0
Outlet_Size                  2410
Outlet_Location_Type            0
Outlet_Type                     0
Item_Outlet_Sales               0
dtype: int64

 '''

 # filling the missing values
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean() , inplace=True )

mode_of_Outlet_Size = big_mart_data.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type' , aggfunc  = (lambda x : x.mode()[0] )) 
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values , 'Outlet_Size'] = big_mart_data.loc[miss_values , 'Outlet_Type'].apply(lambda x: mode_of_Outlet_Size[x])

# print(big_mart_data.isnull().sum())
''' Item_Identifier              0
Item_Weight                  0
Item_Fat_Content             0
Item_Visibility              0
Item_Type                    0
Item_MRP                     0
Outlet_Identifier            0
Outlet_Establishment_Year    0
Outlet_Size                  0
Outlet_Location_Type         0
Outlet_Type                  0
Item_Outlet_Sales            0
dtype: int64 '''


''' sns.set()
plt.figure(figsize = (6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()
plt.figure(figsize = (6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()
plt.figure(figsize = (6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()
plt.figure(figsize = (6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()
plt.figure(figsize = (6,6))
sns.countplot(x='Outlet_Establishment_Year', data =big_mart_data )
plt.show()
plt.figure(figsize = (30,6))
sns.countplot(x='Item_Type', data =big_mart_data )
plt.show()
plt.figure(figsize = (6,6))
sns.countplot(x='Outlet_Size', data =big_mart_data )
plt.show()
 '''


big_mart_data.replace({'Item_Fat_Content' :  {'low fat' :'Low Fat' , 'LF' : 'Low Fat'   , 'reg' : 'Regular' }}, inplace= True )

encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])


# big_mart_data.head()


X = big_mart_data.drop(columns = 'Item_Outlet_Sales' ,axis = 1)
Y = big_mart_data['Item_Outlet_Sales']
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size= 0.2, random_state = 2)
# print(X.shape , X_train.shape ,X_test.shape ) # (8523, 11) (6818, 11) (1705, 11)



regressor = XGBRegressor()
regressor.fit(X_train , Y_train)


training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train , training_data_prediction)
print('R squared value (train) = ',r2_train)


testing_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test , testing_data_prediction)
print('R squared value (test) = ',r2_test)
