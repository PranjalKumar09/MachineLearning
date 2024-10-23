# Regression

Regression is used to predict continuous values based on one or more independent variables.

## Key Concepts:
- **X**: Independent variable(s)
- **Y**: Dependent variable (Target)

---

## Types of Regression:

### 1. Simple Regression:
Involves **one independent variable**.

- **Simple Linear Regression**: A linear relationship between X and Y.
- **Simple Non-linear Regression**: A non-linear relationship between X and Y.

### 2. Multiple Regression:
Involves **more than one independent variable**.

- **Multiple Linear Regression**: A linear relationship between multiple X's and Y.
- **Multiple Non-linear Regression**: A non-linear relationship between multiple X's and Y.

---

## Linear Regression:

The goal is to minimize the **Mean Squared Error (MSE)**.

### Formula:
\[ y = \theta_0 + \theta_1 x_1 \]

Where:
- \(\theta_0\) is the intercept.
- \(\theta_1\) is the slope.

### Calculating the Slope \(\theta_1\):
\[
\theta_1 = \frac{\sum_{i=1}^{s} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{s} (x_i - \bar{x})^2}
\]

### Pros:
- 🚀 Very fast to compute
- ⚙️ No tuning required
- 📚 Easy to understand and interpret


#### Workflow of the Linear Regression Model

##### Step 1: Set Learning Rate & Number of Iterations
- Randomly initialize the weights (\( w \)) and bias (\( b \)) values.
- Initial linear equation: \( y = w \cdot x + b \).

##### Step 2: Build the Linear Regression Equation
-  \[
  y = w \cdot x + b
  \]

##### Step 3: Predict the Target Value (\( y_{\text{pred}} \))

##### Step 4: Update Parameters Using Gradient Descent
- Calculate the **loss function** (mean squared error or similar) and update the weight and bias values using gradient descent:
  \[
  w = w - \alpha \cdot \frac{\partial \text{Loss}}{\partial w}
  \]
  \[
  b = b - \alpha \cdot \frac{\partial \text{Loss}}{\partial b}
  \]


##### Step 5: Repeat Steps 3 and 4
- Until loss function is optimized

##### Final Output
- After the iterations, the model will have the **best weight (\( w \)) and bias (\( b \)) values** that minimize the loss function.

``` python

import pandas as pd
from sklearn.model_selection import train_test_split
from Statics_for_ML import LinearRegression
import  matplotlib.pyplot as plt

# Load dataset
salary_data = pd.read_csv("Datasets/csv/salary_data.csv")

# Check for null values
print(salary_data.head())
print(salary_data.shape)
# if null data deal with it but in this data set no null values 
print(salary_data.isnull().sum())

# Remove rows with NaN values
salary_data = salary_data.dropna()

# Split the data 
X = salary_data.iloc[:, :-1].values  
Y = salary_data.iloc[:, 1].values

print("X ", X)
print("Y ", Y)


X_train , X_test, Y_train , Y_test = train_test_split(X,Y,random_state=2 , test_size= 0.33)

# Instantiate and train the model
model = LinearRegression(learning_rate=0.02, num_of_iterations=1000)
model.fit(X, Y)

# Print the learned parameters
print("Weight:", model.w[0])
print("Bias:", model.b)


test_data_prediction = model.predict(X_test)
print(" Test data prediction = " , test_data_prediction)


# Visualizing 
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_test , test_data_prediction , color = 'blue')
plt.xlabel('Work Experience ')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
```

---

### Regression Error:
In the context of regression, **error** is the difference between the predicted and actual values.

---

## Multiple Linear Regression:

It helps in understanding the effectiveness of **multiple independent variables** on the prediction.

### Formula:
\[ y = \theta_0 + \theta_1 c_1 + \theta_2 c_2 + \dots \]

Or in matrix form:
\[ y = \theta^T c \]

⚠️ Using all features can result in **overfitting**, so it's important to select relevant features carefully.

---

