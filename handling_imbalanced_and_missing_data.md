# Handling Imbalanced Data

## pandas.concat() Function

The `pandas.concat()` function performs concatenation operations along a specified axis, handling optional set logic (union or intersection) of the indexes on the other axes.

### Syntax
```python
concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)
```
### Parameters:

- **objs**: Series or DataFrame objects.
- **axis**: Axis to concatenate along; default is `0`.
- **join**: How to handle indexes on other axes; default is `'outer'`.
- **ignore_index**: If `True`, do not use the index values along the concatenation axis; default is `False`.
- **keys**: Sequence to add an identifier to the result indexes; default is `None`.
- **levels**: Specific levels (unique values) for constructing a MultiIndex; default is `None`.
- **names**: Names for the levels in the resulting hierarchical index; default is `None`.
- **verify_integrity**: Check whether the new concatenated axis contains duplicates; default is `False`.
- **sort**: Sort non-concatenation axis if not aligned when `join` is `'outer'`; default is `False`.
- **copy**: If `False`, do not copy data unnecessarily; default is `True`.


#### Example: Handling an Imbalanced Dataset
``` py
import pandas as pd

# Load dataset
credits_df = pd.read_csv("Datasets/credit_data.csv")

# Check the class distribution
print(credits_df['Class'].value_counts())

# Legitimate and Fraudulent transactions
legit = credits_df[credits_df.Class == 0]
fraud = credits_df[credits_df.Class == 1]

# Undersampling Legitimate transactions to match Fraudulent transactions
legit_sample = legit.sample(n=492)

# Concatenate sampled legit transactions with fraud transactions
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.shape)  # (984, 31)
```
## Handling Missing Values

### Methods:
- **Imputation**: Replacing missing values with meaningful data.
- **Dropping**: Removing rows/columns with missing values (not recommended for small datasets).

### Example: Imputing Missing Salary Values
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv('Datasets/Placement_Dataset.csv')

# Check for missing values
print(dataset.isnull().sum())

# Fill missing 'salary' values with mean salary
dataset['salary'].fillna(dataset['salary'].mean(), inplace=True)

# Visualize the distribution of salary after imputation
fig, ax = plt.subplots(figsize=(8,8))
sns.distplot(dataset['salary'])
plt.show()
```



