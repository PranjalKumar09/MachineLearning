
### Libraries Overview
1. **TensorFlow**
   - Developed by Google.
   - Most widely used for deep learning tasks.

2. **PyTorch**
   - Strong competitor to TensorFlow.
   - Preferred in research for its flexibility and dynamic computation graph.

3. **Keras**
   - User-friendly API for quick prototyping and development.
   - Suitable for beginners due to its simplicity.

### Neural Network Structure in Keras

#### 1. Regression Example
- **Purpose:** Predict continuous values.
- **Output Neurons:** Equal to 1 (single output).

```python
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the model
model = Sequential()

# Number of features
n_cols = data_df.shape[1]

# Adding layers
model.add(Dense(5, activation='relu', input_shape=(n_cols, )))  # First hidden layer
model.add(Dense(5, activation='relu'))  # Second hidden layer
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)

# Make predictions
predictions = model.predict(test_data)
```

#### 2. Classification Example
- **Purpose:** Classify data into categories.
- **Output Neurons:** Equal to the number of categories (4 in this case).

```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Initialize the model
model = Sequential()

# Number of features
n_cols = data_df.shape[1]

# Convert target to categorical
target = to_categorical(target)

# Adding layers
model.add(Dense(5, activation='relu', input_shape=(n_cols, )))  # First hidden layer
model.add(Dense(5, activation='relu'))  # Second hidden layer
model.add(Dense(4, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target, epochs=10)

# Make predictions
predictions = model.predict(test_data)
```

#### **Model Structure (Sequential API)**

- **Sequential Model**: 
  - The Sequential model is a linear stack of layers, where you can add one layer at a time. It's useful for building simple models where the data flows in one direction (from input to output).

#### **Output Neurons**
- **Regression Example (Single Output Neuron)**:
  - In a regression task, you are predicting a continuous value (e.g., price, temperature).
  - **Why 1 Neuron?**: Since the output is a single continuous value, you only need one neuron to represent it.

- **Classification Example (Multiple Output Neurons)**:
  - In classification tasks, you categorize input data into different classes (e.g., classifying images of animals).
  - **Why Equal to Categories?**: The number of output neurons must match the number of categories (classes) you want to predict. Each neuron will output a score (or probability) for its corresponding class.

#### **Loss Functions**
- **Mean Squared Error (MSE)**: 
  - Used in regression tasks. It calculates the average of the squares of the errors, providing a measure of how far the predictions are from the actual values.
  - **Why MSE?**: It is sensitive to large errors, which is often desirable when predicting continuous values.

- **Categorical Crossentropy**:
  - Used in multi-class classification tasks. It measures the difference between the predicted probability distribution (output from the softmax layer) and the true distribution (one-hot encoded labels).
  - **Why Categorical Crossentropy?**: It effectively penalizes the model for wrong classifications and encourages it to predict probabilities that closely match the true labels.

#### **Metrics**
- **Accuracy**:
  - Commonly used in classification tasks to measure how many predictions were correct out of the total predictions made.
  - **Why Use Accuracy?**: It provides a straightforward measure of how well the model is performing overall.

#### **Activation Functions**
- **ReLU (Rectified Linear Unit)**:
  - Activation function for hidden layers. It outputs the input directly if it is positive; otherwise, it outputs zero.
  - **Why ReLU?**: It helps to introduce non-linearity to the model and allows for faster training by mitigating the vanishing gradient problem.

- **Softmax**:
  - Activation function for the output layer in multi-class classification. It converts raw output scores (logits) into probabilities by exponentiating and normalizing the scores.
  - **Why Softmax?**: It ensures that the sum of the output probabilities is equal to 1, making it easier to interpret the output as probabilities for each class.


