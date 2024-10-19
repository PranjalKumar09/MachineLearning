
# Important Coding Snippets for Machine Learning

## Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import nltk
```

## Data Preparation
### Data Encoding
``` python
# Replace diagnosis labels
dataset['diagnosis'].replace({'B': 1, 'M': 0}, inplace=True)

# Replace categorical features in car dataset
car_dataset.replace({
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Seller_Type': {'Dealer': 0, 'Individual': 1},
    'Transmission': {'Manual': 0, 'Automatic': 1}
}, inplace=True)
```

### Splitting Data
``` python
X = credit_card_data.drop(columns='Class')
# These both also have same effect
# X = crdit_card_data.drop(columns='Class')
# X = credit_card_data.drop('Class', axis=1)
Y = credit_card_data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.1, stratify=Y)
```

### Model Accuracy
``` python
# Calculate R² score
score = r2_score(Y_train, training_Data_prediction)
print("R squared error (train data):", score)

# Predict user input
input_data = np.asarray(data).reshape(1, -1)
print("Breast-Cancer is Malignant") if model.predict(input_data)[0] == 0 else print("Breast-Cancer is Benign")
```
## Feature Extraction
### TF-IDF Vectorization
``` python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(data_list)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(feature_vectors)
similarity_score = list(enumerate(similarity[index_of_movie]))
```
#### Bag of Words
- **BOW:** List of unique words in the text corpus.
- **TF-IDF**: Measures word importance in documents.
### TF-IDF Calculation
``` python
# Term Frequency (TF)
TF = (Number of times term t appears in document) / (Number of terms in the document)

# Inverse Document Frequency (IDF)
IDF = log(Number of documents / (Number of documents containing term t))

# TF-IDF
TF-IDF = TF * IDF
```
### Handling Imbalanced Data 
``` python
# Load dataset
credits_df = pd.read_csv("Datasets/credit_data.csv")

# Display class distribution
print(credits_df['Class'].value_counts())

# Under-sampling to balance dataset
legit = credits_df[credits_df.Class == 0]
fraud = credits_df[credits_df.Class == 1]
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.shape)  # (984, 31)
```
### Missing Value Handling 
``` python
# Fill missing values in Big Mart dataset
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
mode_of_Outlet_Size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(lambda x: mode_of_Outlet_Size[x])
```
### Clustering Example 
``` python
# K-Means clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
```
### Text Preprocessing and Stemming
``` python
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

port_stem = PorterStemmer()

def stemming(content):
    stemming_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemming_content = [port_stem.stem(word) for word in stemming_content if word not in stopwords.words('english')]
    return ' '.join(stemming_content)

X = np.array([stemming(content) for content in X])
```
### Movie Recommendation Example
``` python
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
combined_features = ' '.join(movies_data[selected_features].fillna('').agg(' '.join, axis=1))

movie_name = input('Enter your favorite movie name: ')
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)
list_of_all_titles = movies_data['title'].tolist()
find_closed_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_movie = find_closed_match[0]
index_of_movie = movies_data[movies_data['title'] == close_movie].index[0]
similarity_score = list(enumerate(similarity[index_of_movie]))

sorted_similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
print("Movie Suggested For You")
for i in range(int(input("How many movie suggestions do you want? "))):
    print(i + 1, ") ", movies_data.iloc[sorted_similarity_score[i][0]]['title'])

```
### Mail Data Handling
``` python
X = mail_data["Message"]
Y = mail_data["Category"]


# All represent same thing
#X = mail_data.Message
#Y = mail_data.Category
# X = mail_data.iloc[:, mail_data.columns.get_loc('Message')]
# Y = mail_data.iloc[:, mail_data.columns.get_loc('Category')]
# X = mail_data.filter(items=['Message'])
# Y = mail_data.filter(items=['Category'])


input_mail = feature_extraction.transform(["Did you catch the bus? Are you frying an egg? ..."])
```

### Filling Missing Values in Titanic Dataset
``` python
titanic_Data['Age'].fillna(titanic_Data['Age'].mean(), inplace=True)
titanic_Data['Embarked'].fillna(titanic_Data['Embarked'].mode()[0], inplace=True)
```


#### Prediction
``` python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
```

#### for getting null values
``` python
df.isnull().sum()[df.isnull().sum()>0]

df_null = df[df.isnull().sum()[df.isnull().sum()>0].index
]
```


### To get all data of column which contain only string value type
``` py
df_objects = df[df.select_dtypes(include=['object']).columns]
```

``` py
 df = pd.concat([df, pd.get_dummies(df[col], prefix = col)], axis = 1)
```
- `prefix=col` ensures that the new columns are named with the format `col_value` (e.g., `col1_val1`, `col1_val2` are now columns).



#### This is example that column having more than 1100 value are removed 
``` py
df_objects = df_objects.drop(df_objects[df_objects.isna().sum()[df_objects.isna().sum() > 1100].index], axis = 1)
```

#### Dummy Variable Encoding
``` py
# Create dummy variables for categorical columns
df_encoded = pd.get_dummies(df_objects)

# Drop columns with 'null' in their names
# because if all other value will null , then it will come like column1_null
for col in df_encoded.columns:
    if 'null' in col:
        df_encoded = df_encoded.drop(col, axis=1)
        print(f"Dropped column: {col}")
```
#### Combining Encoded Data
Combine the original DataFrame with the encoded variables and drop original categorical columns.
``` py
# Combine the original DataFrame with encoded variables
new_df = pd.concat([df, df_encoded], axis=1)


# Drop original categorical object columns
new_df = new_df.drop(df.select_dtypes(include=['object']), axis=1)
```

#### Handling Missing Values
``` py
# Check for missing values
missing_values = new_df.isna().sum()
print("Missing Values:\n", missing_values[missing_values > 0])

# Define columns for filling missing values
mode_columns = ['Col1_mean_type', 'Col2_mean_type' ... ]
mean_columns = ['Col1_mode_type', 'Col2_mode_type', .. ]

# Fill missing values with mode for categorical columns
for col in mode_columns:
    new_df[col] = new_df[col].fillna(new_df[col].mode()[0])

# Fill missing values with mean for numerical columns
for col in mean_columns:
    new_df[col] = new_df[col].fillna(np.round(new_df[col].mean()))

```




1. **`impute_lis`**: List of columns (`Int_col1`, `Int_col2`, `bool_col1`, `bool_col2`, etc.) that need imputation.

2. **`rest`**: Contains the remaining columns that are **not** in `impute_lis`.

3. **Separate DataFrames**:
   - `df_rest`: Subset of `df` with columns in `rest`.
   - `df_imputed`: Subset
    of `df` with columns in `impute_lis`, where missing values are imputed using **`KNNImputer()`**.

4. **Recombine**:
   - After imputation, `df_rest` and `df_imputed` are concatenated (horizontally) to form the final DataFrame `df`.
``` py
impute_lis = ['Int_col1', 'Int_col2', 'Int_col3', .. , 'bool_col1', 'bool_col2', 'bool_col3'... ]
rest = list(set(df.columns) - set(impute_lis))
df_rest = df[rest]
imp = KNNImputer()
df_imputed = imp.fit_transform(df[impute_lis])
df_imputed = pd.DataFrame(df_imputed, columns = impute_lis)
df = pd.concat([df_rest.reset_index(drop = True), df_imputed.reset_index(drop = True)], axis = 1)
```




#### Drop target column from testing data
We combined in starting so can do all encoding and taking in it along with training data
``` py
testing_data = testing_data.drop(columns='Target_y')
```
####
- **n_estimators** is attribute in both Random Forest & XGB regressor


``` py 
df.corr()['Target'].sort_values(ascending = False)
```
This give effect of each column on Target (-1 to 1)
Then we can do make new columns -> 
``` py
df['3_high_cols'] = df['high_col1'] + df['high_col2'] + df['high_col3']
df['3_low_cols'] = df['low_col1'] + df['low_col2'] + df['low_col3']```