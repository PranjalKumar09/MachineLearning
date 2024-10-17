# Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV

## Parameters

### Model Parameters
- These are internal parameters determined by training the model on the data:
  - **Weight (w)**
  - **Bias (b)**
  
  Formula: `Y = w*X + b`

### Hyperparameters
- External parameters that control the learning process and are adjusted to optimize the model:
  - **Learning Rate**
  - **Number of Epochs**
  - **n_estimators**

## Hyperparameter Tuning
- The process of selecting the best hyperparameters for a machine learning model.
- Also known as **Hyperparameter Optimization**.

### Types of Hyperparameter Tuning
1. **GridSearchCV**: Exhaustive search by evaluating every combination of hyperparameters. Computationally expensive.
2. **RandomizedSearchCV**: Evaluates a random set of hyperparameter combinations. Recommended for large datasets due to lower computational cost.

## Example Code: Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV

```python
import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv("Datasets/csv/data.csv")

# Encode target labels
label_encoder = LabelEncoder()
dataset['diagnosis'] = label_encoder.fit_transform(dataset['diagnosis'])

# Feature matrix (X) and target variable (Y)
X = dataset.drop(['diagnosis', 'Unnamed: 32'], axis=1)
Y = dataset['diagnosis']

# Define model
model = SVC()

# Define hyperparameters to search
parameters = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 5, 10, 20]
}

## GridSearchCV
classifier = GridSearchCV(model, parameters, cv=5)
classifier.fit(X, Y)

# Best hyperparameters and accuracy from GridSearchCV
best_params = classifier.best_params_
highest_accuracy = classifier.best_score_

# Display results
result = pd.DataFrame(classifier.cv_results_)
grid_search_result = result[['param_C', 'param_kernel', 'mean_test_score']]
print(grid_search_result)

## RandomizedSearchCV
classifier = RandomizedSearchCV(model, parameters, cv=5)
classifier.fit(X, Y)

# Best hyperparameters and accuracy from RandomizedSearchCV
best_params = classifier.best_params_
highest_accuracy = classifier.best_score_

# Display results
result = pd.DataFrame(classifier.cv_results_)
random_search_result = result[['param_C', 'param_kernel', 'mean_test_score']]
print(random_search_result)
```