"""
WorkFlow ->

News Data -> Data Pre Processing ->  Train Test Split-> Logisitcs Regression Modeel  

NewData -> Trained Logistics Regression Data 

About the dataset:

id: unique id for a news article 
title: title of the news article
author: author of the news article
text: text of the news article
label: a label that marks whether the news article is real or fake 

1 : Fake News
0 : Real News



"""


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Data Preprocessing
news_data = pd.read_csv("Datasets/csv/fake_or_real_news.csv")
news_dataset = news_data.fillna('')


if 'author' in news_dataset.columns:
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
else:
    news_dataset['content'] = news_dataset['title']

# Separating the data and label 
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Applying stemming to the content column
port_stem = PorterStemmer()
def stemming(content):
    stemming_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemming_content = [port_stem.stem(word) for word in stemming_content if not word in stopwords.words('english')]
    stemming_content = ' '.join(stemming_content)
    return stemming_content

X = np.array([stemming(content) for content in X])

# Transforming data using TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting the data
X_Train, X_test, Y_Train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=12)

# Training the model: Logistic Regression
model = LogisticRegression()
model.fit(X_Train, Y_Train)

# Evaluating
X_train_prediction = model.predict(X_Train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_Train)

print("Accuracy Score of the training data:", training_data_accuracy) # Accuracy Score of the training data: 0.9863581730769231

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy Score of the test data:", testing_data_accuracy)   # Accuracy Score of the test data: 0.9790865384615385

plt.figure(figsize=(10, 5))
plt.bar(['Training Data', 'Test Data'], [training_data_accuracy, testing_data_accuracy], color=['blue', 'orange'])
plt.xlabel('Dataset')
plt.ylabel('Accuracy Score')
plt.title('Model Accuracy Scores')
plt.ylim([0, 1])
plt.savefig('accuracy_scores.png')  #
# plt.show()
plt.close()

# Plotting the distribution of fake vs real news
plt.figure(figsize=(10, 5))
news_dataset['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Fake vs Real News')
plt.xticks(ticks=[0, 1], labels=['Real News', 'Fake News'], rotation=0)
plt.savefig('news_distribution.png') 
# plt.show()
plt.close()




text = news_dataset.iloc[0]['text']
# print("Data-> ", text)


X_new = X_test[0]
prediction = model.predict(X_new)

prediction = model.predict(X_new) # it gives in list 
# print(prediction)




if prediction[0] == 1:
    print("This is a fake news")
else:
    print("This is a real news")
