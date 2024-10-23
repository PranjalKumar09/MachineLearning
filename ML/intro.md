# Major Machine Learning Techniques

- **Regression/Estimation**
- **Classification**
- **Clustering**
- **Association**: Frequent co-occurrence

## Artificial Intelligence
Artificial Intelligence (AI) is a branch of computer science concerned with building smart and intelligent machines.

### Examples
- **Non-Intelligent Machines**: Bike, Watch
- **Intelligent Machines**: Tesla Car, Alexa

## Machine Learning
Machine Learning (ML) is a technique to implement AI that allows machines to learn from data without being explicitly programmed.


## Deep Learning
Deep Learning is a subfield of Machine Learning that uses Artificial Neural Networks to learn from data, akin to the neurons in the brain.

### Limitations of Deep Learning
- Neural Networks require a lot of data
- High computational power is needed
- More complex to implement than traditional ML models
- Training Neural Networks can be time-consuming

## Machine Learning Model
A Machine Learning model is a function that establishes a relationship between features and the target variable. It seeks to identify patterns in data, understand it, and train on it. Based on this learning, the ML model makes predictions and recognizes patterns.

### Types of Learning
1. **Supervised Learning**: Involves labeled data
   - Controlled Environment
   - More method than unsupervised learning
   
   #### Techniques :-
   - **Classification** 
   - **Regression**

   ##### Classification Algorithms
   - Logistic Regression
   - Decision Tree Classification
   - Random Forest Classification
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
   - Neural network


   ##### Regression
   - logistic Regression
   - Polynomial Regression
   - Support Vector Machines (SVM)
   

2. **Unsupervised Learning**: Involves unlabeled data
    - UnControlled Environment
    - Less method than unsupervised learning

3. **Reinforcement Learning**: Involves how an intelligent agent takes actions in an environment to maximize its rewards.
    - **Components**:
        - Environment
        - Agent
        - Action
        - Reward
   - Components: Environment, Agent, Action, Reward
   - Examples: Game Playing Robot, Self-Driving Car

## Types of Unsupervised Learning
- **Clustering**: Groups similar data points (e.g., customer segmentation)
    - Discovering  structure
    - Summarization
    - Anomaly detection

- **Association**: Finds important relationships between data points

## Other technique are
    - Density estimation
    - Market based analysis

### Unsupervised Learning Algorithms
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Apriori

## Dimensionality Reduction
Dimensionality Reduction is a technique used to reduce the number of input variables in a dataset. A higher number of dimensions can affect the efficiency of a model.

### Two Methods of Dimensionality Reduction
1. **Feature Selection**: The process of automatically or manually selecting features that contribute most to the prediction variable or output.
2. **Feature Extraction**: The process of reducing an initial set of data by identifying key features for machine learning.

## Evaluation Metrics for Classification
- **Accuracy Score**: The ratio of the number of correct predictions to the total number of input data points.

  \[
  \text{Accuracy Score} = \left( \frac{\text{Number of Correct Predictions}}{\text{Total Number of Data Points}} \right) \times 100\%
  \]

  ```python
  from sklearn.metrics import accuracy_score

### Examples of Intelligent Machines
- **Virtual Personal Assistants**: Siri, Alexa, Cortana, Google Assistant (use speech recognition to follow directions)
- **Industrial Robots**: Equipped with built-in sensors
- **Fraud Detection**: Debit and credit card fraud detection systems
- **Speech Recognition and Voice Synthesis**: Software on computers
- **Self-Driving Vehicles**: Utilize vision instead of relying solely on road markings
- **Healthcare Diagnostics**
- **Smart Weapons**: Capable of identifying targets



### Types of Algo
#### Lazy Algorithm
##### Take more time on testing

#### Eager Learning
##### Take more time on training