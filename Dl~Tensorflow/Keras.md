### Keras Sequential vs. Functional API

- **Sequential API**: Ideal for linear, single-input, single-output models.  
  ```python
  model = Sequential([
      Dense(64, activation="relu", input_shape=(784,)),
      Dense(10, activation="softmax")
  ])
  ```
  
- **Functional API**: Allows for more complex architectures like multi-branch or multi-output models.
  ```python
  inputs = Input(shape=(784,))
  x = Dense(64, activation="relu")(inputs)
  outputs = Dense(10, activation="softmax")(x)
  model = Model(inputs=inputs, outputs=outputs)
  ```

### Functional API Advantages
1. **Flexible Architectures**: Supports complex models (e.g., multi-branch).
2. **Explicit Structure**: Clear connections between layers.
3. **Reusability**: Components can be modular and reusable.

---

### Functional API Capabilities

#### Handling Multiple Inputs & Outputs
- Enables multiple inputs and outputs, sharing and combining layers flexibly.
  ```python
  inputA = Input(shape=(64,))
  inputB = Input(shape=(128,))
  x = Dense(8, activation='relu')(inputA)
  y = Dense(8, activation='relu')(inputB)
  combined = concatenate([x, y])
  z = Dense(1, activation='linear')(combined)
  model = Model(inputs=[inputA, inputB], outputs=z)
  ```

#### Shared Layers & Complex Architectures
- Shared layers allow for shared processing.
  ```python
  shared_layer = Dense(64, activation='relu')
  outputA = shared_layer(inputA)
  outputB = shared_layer(inputB)
  model = Model(inputs=[inputA, inputB], outputs=[outputA, outputB])
  ```

---

### Keras Model Subclassing API
- **Flexibility**: Define dynamic architectures and custom training loops.
- **Implementation**:
  ```python
  class MyModel(tf.keras.Model):
      def __init__(self, units):
          super(MyModel, self).__init__()
          self.dense = Dense(units)

      def call(self, inputs):
          return self.dense(inputs)
  ```

---

### Layer Enhancements

#### Dropout Layer
- Adds regularization to prevent overfitting.
  ```python
  hidden_layer = Dense(64, activation='relu')(input_layer)
  dropout_layer = Dropout(0.5)(hidden_layer)
  ```

#### Batch Normalization
- Normalizes inputs for faster and stable training.
  ```python
  model.add(Dense(128, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  ```

---

### Custom Layers
- **Purpose**: Incorporate novel algorithms or optimize specific needs.
- **Implementation**:
  ```python
  class MyCustomLayer(Layer):
      def build(self, input_shape):
          self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal')
      def call(self, inputs):
          return tf.matmul(inputs, self.w)
  ```

---

### TensorFlow Ecosystem

- **TensorFlow Lite**: For mobile and embedded deployment.
- **TensorFlow.js**: ML in JavaScript environments.
- **TensorFlow Extended (TFX)**: Manages and deploys production ML pipelines.
- **TensorFlow Hub**: Repository of reusable ML modules.
- **TensorBoard**: Visualization tool for model metrics, graphs, and training data insights. 

Each of these components allows TensorFlow to support the full machine learning lifecycle, from research to deployment