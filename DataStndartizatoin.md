## Data Standardization

**Data Standardization**  the process of converting data to a common format and range. It ensures consistency across different sources and improves data quality, integration, and analysis.

### Importance of Data Standardization in Machine Learning:

- **Improving Convergence**: Algorithms like gradient descent converge faster when features are scaled or standardized, preventing certain features from dominating due to their larger scales.
- **Ensuring Fair Comparison**: Ensures all features contribute equally to the model, avoiding biased results from features with larger scales.
- **Enhancing Interpretability**: Helps in interpreting model coefficients or weights by putting features on the same scale.
- **Improving Numerical Stability**: Enhances stability in computations, especially in algorithms sensitive to large feature variations.
- **Algorithm Assumptions**: Algorithms like KNN and SVM assume standardized features, leading to more reliable results.
- **Facilitating Regularization**: Standardizing helps control regularization uniformly across features.
- **Handling Distance-based Algorithms**: Algorithms relying on distance measures (e.g., K-means, hierarchical clustering) benefit from standardized features.

---

### Breast Cancer Dataset Example (Standardization Workflow)

``` python

# Split data into features (X) and target labels (Y)
X, Y = data.data, data.target

# Train-test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# fit the scaler to training data and transform it
scalar.fit(x_train )
x_train_standartized = scalar.transform(x_train)

x_test_standartized = scalar.transform(x_test)

# print(x_train_standartized.std()) # 1.0
# print(x_test_standartized.std()) # 0.8654 ..

```