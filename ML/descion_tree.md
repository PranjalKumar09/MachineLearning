# Decision Tree Model

- **Supervised Learning Model**
- Used for **Classification & Regression**
- Builds a decision node at each step
- Forms the basis for other tree-based models

### Tree Structure
Decision trees can be represented as either **binary trees** or **multiple trees**. 


### Advantages
- Handles both **Classification & Regression** tasks.
- **Easy to interpret**: Decision paths are simple to understand.
- **No need for normalization or scaling** of data.
- **Not sensitive to outliers** or measurement errors.

### Disadvantages
- **Overfitting**: Can create overly complex trees that don't generalize well.
- **Instability**: Small changes in data can lead to significant tree changes.
- **High training time**, especially for large datasets.

---

## Steps to Build a Decision Tree

1. **Choose an attribute** from your dataset.
2. **Calculate the significance** of the attribute in splitting the data.
3. **Split the dataset** based on the significance of the attribute.

---

## Entropy
- **Entropy** is the quantitative measure of the randomness or uncertainty in the information being processed.
  
  - **High entropy** = high randomness.
  - **Low entropy** = more order in the dataset.

Entropy Formula:
$$
\text{Entropy} = - \sum_{i=1}^{n} p_i \log_2(p_i)
$$
p₁, p₂, ..., pₙ represent the probabilities of the classes in the dataset.

---

## Information Gain
- **Information Gain** measures how much a feature reduces entropy (uncertainty) in the classification.
  
  - **Low entropy** = **Increased Information Gain**.
  - The formula for Information Gain can be found here:  
$$
\text{Information Gain} = \text{Entropy(parent)} - \sum_{i=1}^{k} \frac{n_i}{n} \cdot \text{Entropy}(i)
$$
Where:
n_i is the number of instances in child node i,
n is the total number of instances in the parent node,
k is the number of child nodes.

---

## Gini Impurity
- A split in the decision tree is said to be **pure** if all data points are accurately classified.
- **Gini Impurity** measures the likelihood that a randomly selected data point would be incorrectly classified by a specific node.

$$
\text{Gini Impurity} = 1 - \sum_{i=1}^{n} p_i^2
$$


---

# Creating a Regression Tree

**Regression Trees** are implemented using the `DecisionTreeRegressor` from `sklearn.tree`. Below are the important parameters:

- **criterion**: `{"mse", "friedman_mse", "mae", "poisson"}` - The function used to measure error.
- **max_depth**: The maximum depth of the tree.
- **min_samples_split**: The minimum number of samples required to split a node.
- **min_samples_leaf**: The minimum number of samples that a leaf node can contain.
- **max_features**: `{"auto", "sqrt", "log2"}` - The number of features to consider when looking for the best split.

### Example:

We can start by creating a `DecisionTreeRegressor` object and setting the criterion parameter to "mse" (Mean Squared Error).

```python
from sklearn.tree import DecisionTreeRegressor

regression_tree = DecisionTreeRegressor(criterion='mse')
    
```



