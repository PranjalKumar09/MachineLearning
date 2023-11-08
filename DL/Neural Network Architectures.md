
### Shallow vs. Deep Neural Networks
- **Shallow Neural Network**: Consists of one hidden layer.
- **Deep Neural Network**: Contains multiple hidden layers with many neurons per layer.
- **Advancements**:
  - **Data availability**: More data has driven deeper networks.
  - **Computation power**: Increased computational resources (e.g., GPUs) allow training of deep networks.

---

### Convolutional Neural Networks (CNNs)
- **Overview**: CNNs are neural networks specifically designed to process images, making training more efficient for image-related tasks.
- **Applications**: Image recognition, object detection, and computer vision tasks.
- **Image input format**:
  - Grayscale images: \( n \times m \times 1 \)
  - Colored images: \( n \times m \times 3 \)
  
- **Convolutions**: Dot product operation across the input matrix. Example:
  - Each \(2 \times 2\) section of the matrix is multiplied by:
    \[
    \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix}
    \]
  - Output size is reduced from \( n \times m \) to \( (n-1) \times (m-1) \).

- **Pooling Layer**:
  - **Max pooling**: Takes the maximum value from each \(2 \times 2\) section.
  - **Average pooling**: Averages the values from each \(2 \times 2\) section.
  - Output size is halved: \( n \times m \rightarrow \frac{n}{2} \times \frac{m}{2} \).

- **Fully Connected Layer**: A \(1 \times n\) matrix where every value is connected to all neurons in the next layer.

![Image](../Image/cnn.png)


---

### Recurrent Neural Networks (RNNs)
- **Overview**: RNNs have loops, allowing them to process sequences like handwriting, genomes, and stock market data.
- **Example**: Long Short-Term Memory (LSTM), a popular RNN model, used for:
  1. Image generation
  2. Handwriting generation

---

### Autoencoders
- **Overview**: Autoencoders are unsupervised neural networks used for data compression, learning how to compress and decompress data automatically.
- **Algorithm**: Attempts to reconstruct input \(x\) without needing labels. Learns better data projections than methods like PCA.
  
- **Restricted Boltzmann Machines (RBMs)**: A popular type of autoencoder.
  - **Applications**:
    - Handling imbalanced datasets
    - Estimating missing values
    - Automatic feature extraction

These notes condense the key concepts and advancements related to neural networks, CNNs, RNNs, autoencoders, and RBMs.