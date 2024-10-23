# Data Processing and Feature Engineering

## Importance of Data
*More data leads to better predictions.*

## Data Access
- `news_dataset['label']` returns a Pandas Series.
- `news_dataset['label'].values` returns a NumPy array.

## Libraries
```python
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
```
## Note
To` download NLTK stopwords, uncomment the following line:
``` py
# nltk.download("stopwords")
```
## Loading the Dataset
```
news_data = pd.read_csv("Datasets/train.csv")
# If label is 0 -> Real News, 1 -> Fake News
# Check dataset shape and missing values
# print(news_data.shape)  # (20800, 5)
# print(news_data.isnull().sum())
```
## Handling Missing Values
``` py
# Replace missing values with "null"
news_data = news_data.fillna("null")
```
## Merging Author and Title
``` py
news_data['content'] = news_data['author'] + news_data['title']
```
## Stemming Function
Stemming reduces words to their root form (e.g., "actor" and "actress" → "act").


``` py
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(stemmed_content)

news_data['content'] = news_data['content'].apply(stemming)
```
## Feature Extraction
``` py
# Separate data and label
X = news_data['content'].values
Y = news_data['label'].values

# Convert textual data to feature data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Print transformed features
print(X)

```
## Cancer Diagnosis Data Processing
In the following code, we encode the `diagnosis` column with values "M" (Malignant) and "B" (Benign).
``` py
cancer_data = pd.read_csv('Datasets/data.csv')
# Check the count of different labels
# print(cancer_data['diagnosis'].value_counts())  # B: 357, M: 212

# Load the LabelEncoder function
label_encode = LabelEncoder()
cancer_data['target'] = label_encode.fit_transform(cancer_data['diagnosis'])

```
## Key Points
- Encoding is done alphabetically, starting from 0.
- The encoding works well for binary classifications.


``` py
#feature engineering
bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['mean_amt_spent'] = df[bill_cols].mean(axis = 1)
df['std_amt_spent'] = df[bill_cols].std(axis = 1) 
```
-  We should combine & reduce column like these (like expense) which seem to no effect




### Example

  1. **Pclass_bin_Fare**:
    - **Formula**: `df['Pclass_bin_Fare'] = df['Fare'] // df['Pclass']`
    - **Explanation**: This feature divides the `Fare` by the `Pclass` to understand how much fare is paid per class. It could provide insights into how fare varies across different classes (1st, 2nd, and 3rd).
    - **Use Case**: This feature might indicate whether higher fares are concentrated in higher classes or if lower classes pay disproportionately more for services.
  2. **Pclass_bin_Sex**:
    - **Formula**: `df['Pclass_bin_Sex'] = df['Pclass'] - df['Sex_female']`
    - **Explanation**: This feature subtracts a binary representation of gender (`Sex_female`, where 1 indicates female and 0 indicates male) from `Pclass`. This could help differentiate how gender interacts with class, perhaps indicating whether women were more represented in a particular class.
    - **Use Case**: Analyzing this feature may reveal trends in survival rates, social status, or privilege based on both class and gender.
### General Considerations for Feature Engineering

  - **Domain Knowledge**: Understanding the data and the domain can significantly help in creating meaningful features.
  - **Interaction Terms**: Features that capture the interaction between two or more variables can uncover relationships not apparent when looking at variables individually.
  - **Scaling**: Features might need to be scaled or normalized to ensure that they contribute equally to the model's learning.
  - **Handling Categorical Variables**: Categorical variables may need to be encoded (e.g., one-hot encoding) to be used effectively in algorithms that require numerical input.
### Example Feature Engineering Techniques

  - **Binning**: Grouping continuous variables into discrete bins.
  - **Polynomial Features**: Creating polynomial combinations of existing features.
  - **Encoding**: Transforming categorical variables into numerical formats.
  - **Aggregating**: Summarizing data points to create features (e.g., mean, median).
  - **Temporal Features**: Extracting parts of datetime variables (e.g., year, month, day) for models that can benefit from time-related insights.









### **Sample**  
A **sample** is a single instance or example of real-world data used for analysis or training in machine learning. 

**Example:**  
- In image processing, each image is a **sample**.  
- The **features** of the sample are the individual pixel values that represent different aspects of the image.

---

### **Preprocessing by Data Type**

Different types of data require specific preprocessing techniques before being fed into machine learning models. Below are common data types and the preprocessing steps typically applied to each:

| **Input Type**  | **Preprocessing Needed** |
|-----------------|--------------------------|
| **Numeric**     | **Centering & Scaling**: Numeric data is often normalized or standardized by centering (subtracting the mean) and scaling (dividing by the standard deviation) to ensure all features contribute equally to the model. |
| **Categorical** | **Integer Encoding**: Categorical values are converted to integers representing distinct categories (e.g., 'red' → 1, 'blue' → 2). <br> **One-Hot Encoding**: Converts categorical variables into binary vectors (e.g., 'red' → [1,0], 'blue' → [0,1]). This avoids implying any ordinal relationship between categories. |
| **Text**        | **TF-IDF (Term Frequency-Inverse Document Frequency)**: Measures the importance of a word in a document relative to a corpus, emphasizing unique terms. <br> **Embeddings**: Converts words or phrases into dense vectors in a continuous space, capturing semantic relationships (e.g., word2vec, GloVe). |
| **Image**       | **Pixel Representation (RGB)**: Images are typically represented as a 2D matrix of pixel values for grayscale, or 3D for RGB (Red, Green, Blue) channels, where each pixel contains intensity values between 0-255. |
| **Speech**      | **Time Series Representation**: Speech is represented as a time series of numerical values, often capturing sound waves at various time intervals. Preprocessing may involve techniques like **MFCC (Mel Frequency Cepstral Coefficients)**, which helps extract frequency-based features from audio data. |
