

Tensorflow capablities for sequential network

Layers ->
    RNNs
    LSTM
    GRUs (Gated Recurrent Unit)
    Convolutional layers for Sequence data (Conv1d)
    Used for time series

Handling Text Data with TensorFlow:
    Tokenization & padding
    Text Vectorization layer: Convert text data into numerical vectors


model_rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

# Compile and train the model
model_rnn.compile(optimizer='adam', loss='mse')
history_rnn = model_rnn.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Make predictions
predictions_rnn = model_rnn.predict(X_test)

# Plot the results
plt.plot(y_test, label='True Data')
plt.plot(predictions_rnn, label='RNN Predictions')
plt.legend()
plt.title("RNN Predictions vs. True Data")
plt.show()

from tensorflow.keras.layers import TextVectorization

# Sample text data
texts = ["TensorFlow is great for sequential data tasks.",
         "RNNs and LSTMs are powerful tools for time series.",
         "Text preprocessing involves tokenization and padding."]

# Define TextVectorization layer
vectorizer = TextVectorization(output_sequence_length=10)

# Adapt the vectorizer to the text data
vectorizer.adapt(texts)

# Transform the text data into numerical format
text_vectorized = vectorizer(texts)

print("Vocabulary:", vectorizer.get_vocabulary())
print("Tokenized and padded text sequences:\n", text_vectorized.numpy())
