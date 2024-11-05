# Model Evaluation

## Data Preprocessing Pipeline
When working with datasets that contain missing values, it's crucial to handle them appropriately to avoid errors during model training.

### Workflow:
Data -> Data Preprocessing -> Data Anaysis -> Train-Test Split -> Machine Learning Model -> Evaluation


### Splitting the Data:
```python
X_train, X_Test, Y_train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_Test.shape)  # (768, 8) (614, 8) (154, 8)
print(Y.shape, Y_train.shape, Y_Test.shape)  # (768,) (614,) (154,)
```


### Training Accuracy:
- Higher training accuracy can lead to overfitting, where the model becomes overly tuned to the training dataset. This may result in the model capturing noise and producing a non-generalized version.
- To prevent overfitting, we can use techniques like train-test-split or cross-validation.


### Out-of-Sample Accuracy:
It refers to how well a machine learning model performs on unseen data that was not used during training. A common method to assess this is **K-fold cross-validation**


### K-fold cross-validation
- It involves splitting the dataset into **"K"** number of folds. Each time, one fold is used as test data while the remaining folds are used for training. The process is repeated, with a different fold acting as the test set each time, and the results are averaged.

#### Advantages of K-fold Cross Validation:
1. Better alternative to train-test-split, especially when the dataset is small.
2. More reliable for multiclass classification problems.
3. Provides a more robust evaluation of the model's performance.
4. Useful for model selection.

### Model Comparison Using Train-Test Split:
``` python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Splitting the dataset
X = dataset.drop(columns='target', axis=1)
Y = dataset['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3, stratify=Y)

# Model list
models = [
    LogisticRegression(max_iter=1000),
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier()
]

# Function to compare models using train-test split
def compare_models_train_test():
    for model in models:
        model.fit(X_train, Y_train)
        test_data_predictions = model.predict(X_test)
        accuracy = accuracy_score(Y_test, test_data_predictions)
        print(f"Accuracy score of the {model} is {accuracy}")

# Uncomment to run the comparison
# compare_models_train_test()
```

### Model Comparison Using Cross-Validation:
``` python
cv_score = cross_val_score(LogisticRegression(max_iter=1000), X, Y, cv=5)
# print(cv_score)

# Mean accuracy across all folds
mean_accuracy = round((sum(cv_score) / len(cv_score)) * 100, 2)
print(mean_accuracy)
```
### Cross-Validation for Multiple Models:
``` python
def compare_models_cross_validation():
    for model in models:
        cv_score = cross_val_score(model, X, Y, cv=5)
        mean_accuracy = round((sum(cv_score) / len(cv_score)) * 100, 2)
        print(f"Accuracy score of the {model} is {mean_accuracy}")
        print("=" * 20)

# Compare models using cross-validation
compare_models_cross_validation()
```
## Evaluation Metrics

### 1. MAE (Mean Absolute Error)
Measures the average of the absolute differences between predicted and actual values.

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

### 2. MSE (Mean Squared Error)
Measures the average of the squared differences between predicted and actual values.

``` python
from sklearn,metrics import mean_squared_error
```

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

### 3. RMSE (Root Mean Squared Error)
The square root of MSE, providing a more interpretable error rate.

\[
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

### 4. RAE (Relative Absolute Error)
A normalized version of the MAE.

\[
\text{RAE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i - \bar{y}|}
\]

where \(\bar{y}\) is the mean of the actual values.

### 5. RSE (Relative Squared Error)
A normalized version of the MSE.

\[
\text{RSE} = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\]

### 6. R² (R-squared)
Indicates how well the model fits the data. The higher the R², the better the model's accuracy.

\[
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
\]

where:
- \(\text{SS}_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\) (Residual sum of squares)
- \(\text{SS}_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2\) (Total sum of squares)



## Evaluation Metrics in Classification

### 1. Jaccard Index
The **Jaccard Index** measures the similarity between the predicted and actual values by comparing their intersection and union.

Formula:

\[
f(y, \hat{y}) = \frac{| y \cap \hat{y} |}{ | y | + | \hat{y} | - | y \cap \hat{y} | }
\]

Where:
- \( y \): actual values
- \( \hat{y} \): predicted values

The Jaccard Index can be interpreted as the percentage of overlap between the predicted and actual values. A higher value indicates better performance.

### 2. F1 Score in Confusion Matrix
The **F1 score** combines **Precision** and **Recall** into a single metric, providing a balanced view of classification performance.

- **Precision**: Measures how many of the predicted positive values are actually positive.

  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

- **Recall (Sensitivity)**: Measures how many of the actual positive values are correctly predicted.

  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

- **F1 Score**: The harmonic mean of Precision and Recall, which balances the two metrics. It is particularly useful when you have an uneven class distribution.


|               | Predicted Positive (y^) | Predicted Negative (y^) |
|---------------|--------------------------|--------------------------|
| **Actual Positive (y)** | True Positive (TP)          | False Negative (FN)         |
| **Actual Negative (y)** | False Positive (FP)         | True Negative (TN)          |

### Definitions:
- **True Positive (TP)**: The actual positive is correctly predicted as positive.
- **False Positive (FP)**: The actual negative is incorrectly predicted as positive (also known as a Type I error).
- **False Negative (FN)**: The actual positive is incorrectly predicted as negative (also known as a Type II error).
- **True Negative (TN)**: The actual negative is correctly predicted as negative.



  \[
  F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

This score is computed for each class, and the **average accuracy** is the average of the F1 scores across all classes.

### 3. Log Loss
**Log Loss** (also known as **Logarithmic Loss** or **Cross-Entropy Loss**) measures the performance of a classifier where the predicted output is a probability between 0 and 1. Unlike accuracy, log loss takes into account the uncertainty of the prediction.

Formula:

\[
\text{LogLoss} = -\left( Y \cdot \log(\hat{Y}) + (1 - Y) \cdot \log(1 - \hat{Y}) \right)
\]

Where:
- \( Y \): actual binary label (0 or 1)
- \( \hat{Y} \): predicted probability (ranging between 0 and 1)

Log Loss penalizes false classifications more heavily if the classifier is confident about its incorrect prediction, making it a useful metric when working with probabilistic classifiers.
