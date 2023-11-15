### TensorFlow Overview

- **Open-source library by Google** for numerical computation and machine learning, optimized for tasks like deep neural networks.
- **Backend**: Built in C/C++ for high performance.
- **Data Flow Graph**: Operations are nodes, and data (multi-dimensional arrays) flows through edges.

---

### Why TensorFlow?

- Supports **Python** and **C++ APIs**.
- **Faster compile times**, ideal for research and deployment.
- **Scalable** across **CPUs, GPUs**, and distributed systems.
- Extensive **mathematical support** for complex tasks.

---

### Key Concepts

- **Data Flow Graph**:
  - **Node**: Mathematical operation (e.g., addition, multiplication).
  - **Edge**: Multi-dimensional array (Tensor).

- **Tensors** can be **multi-dimensional arrays**: 
  - **0D**: Scalar
  - **1D**: Vector
  - **2D**: Matrix
  - Example:
    ```python
    scalar = tf.constant(2)
    vector = tf.constant([5, 6, 2])
    matrix = tf.constant([[1, 2, 3], [2, 3, 4]])
    ```

- **tf.placeholder()**: Used for input matrices in TensorFlow 1.x.
- **tf.Variable()**: Stores model weights that can change during training.

---

### TensorFlow Architecture

- **Python frontend**: User-friendly interface for model building.
- **Core TensorFlow execution system**: Handles tensor operations efficiently.
- **Device support**: Runs on **CPU, GPU**, and mobile devices (Android, iOS).

---

### TensorFlow 2.x Features

- **Keras**: Integrated as the default high-level API.
- **Eager Execution**: Activated by default, allowing immediate execution of operations without building graphs.
  - Example:
    ```python
    scalar = tf.constant(2)
    print(type(scalar))  # Output: EagerTensor
    ```

- **Performance optimizations**: Improved GPU support and efficient execution.
- **Rich ecosystem**: Include wide range of tools & libraries that enhance capabilities


---

### TensorFlow Basic Commands

- **Version Check**:
  ```python
  print(tf.__version__)
  ```

- **Creating Tensors**:
  ```python
  a = tf.constant(5)
  b = tf.Variable([1, 2, 3], dtype=tf.float32)
  ```

- **Basic Operations**:
  - Addition:
    ```python
    result = tf.add(a, b)
    ```
  - Multiplication:
    ```python
    result = tf.multiply(a, b)
    ```

---

### Model Building Example

- **Create a Sequential Model**:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(32, activation='relu', input_shape=(10,)),
      Dense(64, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  ```

- **Compile the Model**:
  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

- **Train the Model**:
  ```python
  history = model.fit(X_train, y_train, epochs=10, batch_size=32)
  ```

- **Evaluate the Model**:
  ```python
  loss, accuracy = model.evaluate(X_test, y_test)
  ```

- **Save/Load the Model**:
  ```python
  model.save('model.h5')
  model = tf.keras.models.load_model('model.h5')
  ```

---

### TensorFlow Advanced Concepts

- **Custom TensorFlow Functions**:
  ```python
  @tf.function
  def custom_op(x, y):
      return tf.add(x, y)
  ```

- **Automatic Differentiation**:
  ```python
  x = tf.Variable(3.0)
  with tf.GradientTape() as tape:
      y = x ** 2
  grad = tape.gradient(y, x)
  ```



### model.summary()
  Tells the layers used in model
