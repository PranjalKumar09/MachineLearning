# Overview of Statistics

Statistics is the science of collecting, analyzing, and presenting data.

### Key Measures:
- **Range**: Difference between max and min values.
- **Mean**: Average value of a dataset.
- **Standard Deviation**: Measure of data spread from the mean.

## Six Sigma
A disciplined approach to process improvement with five steps:
1. **Define** objectives.
2. **Measure** relevant metrics.
3. **Analyze** data for insights.
4. **Improve** based on findings.
5. **Control** to ensure continuous monitoring.

---

## Data Types
### 1. **Categorical Data** 
   - **Nominal**: Classification without quantitative value (e.g., gender, color).
   - **Ordinal**: Data with a natural order (e.g., rankings).
   
### 2. **Numerical Data**
   - **Discrete**: Finite values (e.g., number of students).
   - **Continuous**: Infinite possible values (e.g., weight, height).

---

## Types of Statistics
### 1. **Descriptive Statistics**: Summarizes data.
   - **Measures of central tendency**: Mean, Median, Mode.
   - **Measures of variability**: Range, Standard Deviation, Variance.

### 2. **Inferential Statistics**: Draws conclusions from a sample of data.

---

## Statistical Studies
- **Sample Study**: Analysis on a sample subset.
- **Observational Study**: Observes and analyzes data without intervention.
- **Experimental Study**: Manipulates variables to determine effects.

## Sampling Techniques:
1. **Simple Random Sampling**: Every member has an equal chance.
   - **Pros**: Reduces bias, simple, balanced sample.
   - **Cons**: May not represent the population.
   
2. **Systematic Sampling**: Every nth member is selected.
   - **Pros**: Quick, less bias.
   - **Cons**: Risk of manipulation.

3. **Stratified Random Sampling**: Population divided into strata, then randomly sampled.
   - **Pros**: High precision.
   - **Cons**: Difficult when groups can't be clearly defined.

4. **Cluster Sampling**: Population divided into clusters, some clusters randomly selected.
   - **Pros**: Fewer resources, reduces variability.
   - **Cons**: May not represent the whole population.

---

## Measures of Central Tendency
- **Mean**: Average value of a dataset.
- **Median**: Middle value.
- **Mode**: Most frequent value.

### Measures of Variability
- **Range**: Max value - Min value.
- **Variance**: Average squared deviation from the mean.
- **Standard Deviation**: Square root of variance, indicating data spread.

---

## Percentile and Quantile
- **Percentile**: Value below which a certain percentage of data falls.
- **Quantile**: Divides data into equal-sized subgroups.

---

# Vectors and Operations

### Vector Basics:
- **1D Vectors**: Can be a row or a column.
- **Addition/Subtraction**: Combine corresponding elements.
  - Example: `[1, 3] + [2, 4] = [3, 7]`.
- **Scalar Multiplication**: Enlarge or shrink vectors.

### Vector Similarity:
- Smaller angle between vectors indicates similarity.

### Dot Product:
- Result is a scalar: `np.dot(a, b)`.
- Example: Dot product of `[2, 3]` and `[5, -6]` = `2*5 + 3*(-6) = -8`.

### Cross Product:
- Result is a vector: `np.cross(a, b)`.
- Example: Cross product of `[1, 2, 3]` and `[5, 4, 4]` = `[-8, -11, 6]`.

### Magnitude of a Vector:
```python
magnitude = np.sqrt(np.sum(vector**2))
```

### Projection of Vector a onto b:
``` python
projection = (np.dot(a, b) / magnitude_of_b) * b
```



### Defining a 3x3 Matrix
```python
import numpy as np

matrix_2 = np.array([[10, 35, 45], 
                     [50, 64, 80], 
                     [20, 15, 90]])

# Check the shape of the matrix
print(matrix_2.shape)  # Output: (3, 3)
```
Generate a 2x3 matrix with random floats between 0 and 1.
```python
random_matrix = np.random.rand(2, 3)

# Check the shape and display the random matrix
print(random_matrix.shape)  # Output: (2, 3)
print(random_matrix)        # Displays the random matrix
```
Generate a 2x3 matrix with random integers between 0 and 99.
``` python 
random_matrix_int = np.random.randint(100, size=(2, 3))

# Display the random integer matrix
print(random_matrix_int)

```