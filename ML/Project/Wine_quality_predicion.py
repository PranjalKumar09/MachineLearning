""" 
Wine -> Data analysis -> data pre processing -> Train test split -> random forestx   model 
                                                                              |  
                                                                              |  
                                New dat            ---------------------> Trained random forest model -------------------> Wine quality 


randome forest algorithm :   like binary tree it act on random forest algorithm  
Model training : Random Forest Classifier



"""


from statistics import correlation
import numpy as np
import matplotlib .pyplot as  plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




# Data Collection 
wine_dataset = pd.read_csv("/media/pranjal/New Volume/CODES/Python/Datasets/winequality-red.csv") 

# print (wine_dataset.shape) # (1599, 12)


# print(wine_dataset.isnull().sum())
""" fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: i nt64 """



# data analysis and visulazation



# number of values for each quality 
# sns.catplot(x = 'quality',data= wine_dataset  , kind = 'count' )
# plt.figure ( figsize = (5,5))
# sns.barplot(x = 'quality', y = 'volatile acidity', data = wine_dataset )
# plt.show()

# correlation 
# 1) Poistive correlation 2) Negative Corelation 

# correlation = wine_dataset.corr()

# constructing = heatmap to understand the correlation between the columns 
# plt.figure ( figsize = (10,10))
# sns.heatmap(correlation, annot = True , cbar= True , square= True , fmt = '.1f' , annot_kws = {'size':8} , cmap = 'Blues')
# plt.show()
""" 
    correlation: This is the data for which the heatmap is to be generated. It could be a correlation matrix or any 2D array-like data.

    annot=True: If True, write the data values in each cell.

    cbar=True: If True, draw the color bar on the side of the heatmap.

    square=True: If True, set the Axes aspect to be equal, creating a square-shaped plot.

    fmt='.1f': String formatting code to use when adding annotations. In this case, it formats the numbers with one decimal place.

    annot_kws={'size': 8}: Additional keyword arguments for annotations. In this case, it specifies the size of the annotation text.

    cmap='Blues': Colormap to be used for mapping the data values to colors. 'Blues' is a sequential colormap from light to dark blue.
 """
 
 
"""  data preprocessing seprate the data and label  """
X = wine_dataset.drop(columns= 'quality')
Y = wine_dataset['quality'].apply(lambda y_value : 1 if y_value >= 7 else   0 )

""" now we will have done label binarization or label encoding   in Y"""



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# print(Y.shape, Y_train.shape, Y_test.shape) # 1599,) (1279,) (320,)



model = RandomForestClassifier()
model.fit(X_train , Y_train)



# accuracy_score 

X_test_prediction = model.predict(X_test)
test_data_predicition = accuracy_score(X_test_prediction, Y_test)

# print("Accuracy Score of the test data : ", test_data_predicition) # Accuracy Score of the test data :  0.91875




# Building a Preditive System changing the input data to a numpy arr)
input_data = np.asarray((7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0))

# reshape the data as we are preocessing the label for only one instance
input_data_reshaped = input_data.reshape(1,-1)

#    1: Specifies that the resulting array should have one row.
#   -1: Acts as a placeholder, letting NumPy automatically determine the number of columns needed to preserve the total number of elements in the original array.

prediction = model.predict(input_data_reshaped)
# print(prediction)  # [1]


print("Good Quality Wine ") if prediction[0] else print ("Bad Quality")




