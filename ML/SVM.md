# Support Vector Machine (SVM)

## Overview
Support Vector Machine (SVM) is a **supervised learning** algorithm used for **classification and regression** tasks. It excels in classification by finding the optimal **hyperplane** that separates data points in an N-dimensional feature space. High Dimensoinal feature space

### Key Concepts
- **Hyperplane**: A line or plane that separates data points into different classes. In 2D, it's a line; in 3D, a plane.
- **Support Vectors**: Data points closest to the hyperplane. They influence the position of the hyperplane.

### Advantages
- Effective for smaller datasets.
- Works well with high-dimensional data.
- Efficient when there's a clear margin of separation between classes.

### Disadvantages
- Slow for large datasets.
- Struggles with noisy datasets with overlapping classes.

## Hyperplane Equation
\[
y = wx + b
\]
Maximize the margin between classes:
\[
\frac{2}{||w||}
\]
Minimize error with the equation:
\[
\frac{2}{||w||} + c \sum{e_i}
\]
Where `c` is the number of errors, and `e_i` is the error magnitude.

## SVM Kernels
Kernels transform data to higher dimensions for non-linear classification. Types include:

- **Linear**: \( K(x_1, x_2) = (x_1^T)(x_2) \)
- **Polynomial**: \( K(x_1, x_2) = ((x_1^T x_2) + r)^d \)
- **Radial Basis Function (RBF)**: \( K(x_1, x_2) = \exp(-\gamma ||x_1 - x_2||^2) \)
- **Sigmoid**: \( K(x_1, x_2) = \tanh(\gamma (x_1^T x_2) + r) \)

## Loss Function
The **Hinge Loss** function is used in SVM classifiers:
\[
L = \max(0, 1 - y_i (w^T x_i + b))
\]
Where `b` is the bias and `w` is the weight. It penalizes misclassifications and ensures a large margin from the decision boundary.

## Gradient Descent for Optimization
Gradient Descent minimizes the cost function to find optimal weights (`w`) and bias (`b`):
\[
w_2 = w_1 - \eta \frac{\partial J}{\partial w}
\]
\[
b_2 = b_1 - \eta \frac{\partial J}{\partial b}
\]
Where `η` is the learning rate, and \( \frac{\partial J}{\partial w} \) and \( \frac{\partial J}{\partial b} \) are partial derivatives.

## Use Cases
- Image recognition
- Text classification
- Spam detection
- Sentiment analysis
- Gene expression classification
- Outlier detection, clustering, and regression

## SVM Classifier Example in Python

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM_classifier:
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
    
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for _ in range(self.no_of_iterations):
            self.update_weights()
    
    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)
        for idx, x_i in enumerate(self.X):
            condition = y_label[idx] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - y_label[idx] * x_i
                db = y_label[idx]
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    
    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        return np.where(predicted_labels <= -1, 0, 1)

if __name__ == '__main__':
    data = pd.read_csv('Datasets/csv/diabetes.csv')
    feature = data.drop(columns='Outcome')
    target = data['Outcome']

    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)

    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.2, random_state=2)

    classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
    classifier.fit(X_train, Y_train)

    X_train_prediction = classifier.predict(X_train)
    print('Training Accuracy:', accuracy_score(Y_train, X_train_prediction))

    X_test_prediction = classifier.predict(X_test)
    print('Test Accuracy:', accuracy_score(Y_test, X_test_prediction))
```

In SVM , it is recommended to must have  scaler, it is easier ecludian distance

``` python
scalar = StandardScaler()
scalar.fit(X) 
```


``` python
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standaritze hte input data 
std_data = scalar.transform(input_data_reshaped)

print(std_data)

prediction = classifer.predict(std_data)

print("The Person is Diabiatic ") if prediction[0] == 1 else print("The Person is not Diabiatic ")
```