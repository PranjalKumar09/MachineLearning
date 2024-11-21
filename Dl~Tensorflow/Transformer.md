## Transformer Architecture

### Overview
The Transformer model is a powerful architecture widely used for tasks in **image processing** and **time series prediction**, with prominent implementations such as **BERT** and **GPT**. It operates on the principle of attention mechanisms to capture relationships between elements in sequential data, making it effective for both natural language processing (NLP) and forecasting tasks.

### Key Components

1. **Architecture Structure**:
   - **Transformer Model**:
     - **Encoder** → **Hidden Layer** → **Decoder**
   
2. **Self-Attention Mechanism**:
   - Weighs the importance of input elements (words) and captures dependencies.
   - Enables each input word to attend to every other word, facilitating context and relationship modeling.
   - **Attention Scores**: Calculated as the dot product of Query and Key vectors.

3. **Encoder Composition**:
   - **Multi-Head Self-Attention Layer**: Processes inputs through multiple attention heads for richer representations.
   - **Feedforward Neural Network Layer**: Applies transformations to the output of the self-attention layer.
   - **Residual Connections**: Help mitigate the vanishing gradient problem.
   - **Layer Normalization**: Ensures stable and efficient training.

### Applications
Transformers effectively handle sequential data by capturing long-range dependencies, crucial in:
- **Natural Language Processing**: Text understanding and generation.
- **Time Series Data**: Forecasting future values based on historical patterns.
- **Multimedia Data**: Analyzing audio and video sequences.

### Advantages
Transformers improve upon RNNs and LSTMs by utilizing a self-attention mechanism, enabling parallelization and better management of dependencies across long sequences.

---

### Transformer Implementation


```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

# Attention Layer
class AttentionLayer(Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, queries, keys, values, mask=None):
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(float(self.d_model))
        if mask is not None:
            scores += (mask * -1e9)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attention_weights, values), attention_weights

# Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        queries = self.split_heads(self.wq(x))
        keys = self.split_heads(self.wk(x))
        values = self.split_heads(self.wv(x))
        attention_output, _ = AttentionLayer(self.depth)(queries, keys, values)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        return self.dense(tf.reshape(attention_output, (tf.shape(attention_output)[0], -1, self.num_heads * self.depth)))

# Feed Forward Network Layer
class FeedForwardNetwork(Layer):
    def __init__(self, d_model, dff):
        super().__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))

# Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x)
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output, training=training))

# Encoder Class
class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training):
        for layer in self.enc_layers:
            x = layer(x, training)
        return x

# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.final_layer = Dense(input_vocab_size)

    def call(self, x, training):
        enc_output = self.encoder(x, training)
        return self.final_layer(enc_output)

# Example Usage
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 10000
dropout_rate = 0.1

transformer_model = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
dummy_input = tf.random.uniform((1, 10, d_model))
output = transformer_model(dummy_input, training=False)
print(output.shape)  # Output shape
```

### Transformer Architecture Applications

Transformers are versatile architectures applied across domains, effectively handling long-range dependencies through self-attention.

---

**1. Computer Vision (Vision Transformers)**  
   - **Method**: Images are divided into fixed-size patches, treated as sequences for attention mechanisms.
   - Vision transformers (ViTs) process image patches to classify and detect features, competing with CNNs on image recognition tasks.

---

**2. Speech Recognition (Speech Transformers)**  
   - **Method**: Audio segments are treated as sequence tokens to model temporal dependencies.
   - Speech transformers enable real-time transcription by analyzing contextual patterns within audio sequences.

---

**3. Reinforcement Learning (Decision Transformers)**  
   - **Method**: Models sequences of past actions and rewards for optimal decision-making.
   - Decision transformers apply reinforcement learning to sequentially optimize actions in dynamic environments.

---

``` py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, LayerNormalization, MultiHeadAttention, Dropout, Embedding
from tensorflow.keras.models import Model

# ---------------------------
# Patch Embedding Layer (Used in Vision and Speech Transformers)
# ---------------------------
class PatchEmbedding(Layer):
    def __init__(self, patch_size, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = Dense(d_model)

    def extract_patches(self, x):
        batch_size, height, width, channels = x.shape
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dim = x.shape[-1]
        x = tf.reshape(x, (batch_size, -1, patch_dim))  # Flatten patches
        return x

    def call(self, x):
        patches = self.extract_patches(x)
        return self.projection(patches)

# ---------------------------
# Vision Transformer (ViT) Class
# ---------------------------
class VisionTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, patch_size):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        self.classification_head = Dense(num_classes)

    def call(self, x):
        x = self.patch_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.classification_head(tf.reduce_mean(x, axis=1))  # Global average for classification

# ---------------------------
# Speech Transformer Class
# ---------------------------
class SpeechTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes):
        super(SpeechTransformer, self).__init__()
        self.embedding = Dense(d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        self.classification_head = Dense(num_classes)

    def call(self, x):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.classification_head(tf.reduce_mean(x, axis=1))

# ---------------------------
# Decision Transformer Class (for Reinforcement Learning)
# ---------------------------
class DecisionTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, action_space):
        super(DecisionTransformer, self).__init__()
        self.state_embedding = Dense(d_model)
        self.action_embedding = Dense(d_model)
        self.reward_embedding = Dense(d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        self.policy_head = Dense(action_space)

    def call(self, states, actions, rewards):
        state_embeds = self.state_embedding(states)
        action_embeds = self.action_embedding(actions)
        reward_embeds = self.reward_embedding(rewards)
        
        x = state_embeds + action_embeds + reward_embeds
        for layer in self.encoder_layers:
            x = layer(x)
        return self.policy_head(tf.reduce_mean(x, axis=1))

# ---------------------------
# Attention Layer (used in Encoder)
# ---------------------------
class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout(attn_output, training=training)
        return self.norm(x + attn_output)

# ---------------------------
# Feed Forward Network (FFN) Layer
# ---------------------------
class FeedForwardNetwork(Layer):
    def __init__(self, d_model, dff):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))

# ---------------------------
# Encoder Layer Class
# ---------------------------
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)

    def call(self, x, training):
        attn_output = self.self_attention(x, training=training)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)

```
**Time Series Data**

- **Definition**: A sequence of data points recorded at successive time intervals, often analyzed to understand patterns and make predictions.

- **Traditional Models**:
  - **ARIMA**: Statistical model for forecasting using autocorrelation. Predict future data points by combining three components: autoregression, differencing, and moving averages.
  - **RNNs, LSTMs**: Neural networks specialized for sequential data, capturing dependencies over time.

- **Transformers for Time Series**:
  - Capture long-term dependencies via **self-attention**, enabling insights across lengthy sequences.
  - Support **parallelization**, reducing training time and computational needs.
  - **Adaptable** to variable-length sequences and resilient to missing data.




#### MultiHeadAttention 
- **Purpose**: Applies multiple attention heads to focus on different parts of a sequence in parallel.
- **Key Benefit**: Captures complex relationships, improving context and efficiency.
- **Process**: Multiple attention heads process the input, then combine into a single output.
- **Advantage**: Enhances model’s ability to understand both short- and long-term dependencies.

- : It computes the attention scores and weighted sum of the value
- `def attention(self, query, key, value)`: It computes the attention scores and weighted sum of the values.
