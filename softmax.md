# Activation Function

## Overview
- Determines which nodes propagate information to the next layer.
- Filters and normalizes data.
- Converts output to a nonlinear form.
- Critical for learning complex patterns.

## Common Activation Functions

| Activation Function | Output Description                                      |
|---------------------|---------------------------------------------------------|
| **Sigmoid**          | Outputs values between 0 and 1                          |
| **Tanh**             | Outputs values between -1 and 1                         |
| **ReLU (Rectified Linear Unit)** | Outputs 0 if x < 0, otherwise outputs x      |
| **Softmax**          | Produces a vector of probabilities, where the sum = 1   |

# Softmax Regression

### Multi-class Classification
Softmax regression is used for multi-class classification, implemented in two approaches:

1. **One-vs-All** (OvA)
   - For **K** classes, **K** two-class classifiers are trained.
   - Each classifier separates one class from the others.

2. **One-vs-One** (OvO)
   - Splits the dataset into pairs of classes.
   - For **K** classes, **K(K-1)/2** classifiers are trained.
   - Each classifier distinguishes between two specific classes.

### Softmax & Logistic Regression
- **Logistic Regression** can be extended to **Softmax Regression** for multi-class problems.
- **Softmax** cannot work with Support Vector Machines (SVM).

