"""
Recommadation System
    1) Content Based Recommedation System
    2) Popularity Based Recommedation System
    3) Collaborative Recommedation System 

Work Flow : Data -> Data Preprocessing -> Features extraction -> User Input -> Cosine Similarity -> List of movies

Important columns =>  ['genres' , 'keywords' , 'tagline' , 'cast' , 'director']


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(data_list)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(feature_vectors)
similarity_score = list(enumerate(similarity[index_of_movie]))




"""

import numpy as np 
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv("Datasets/movies.csv")
# print(movies_data.shape) # (4803,24)

# print(movies_data.isnull().sum()) # this files has null values 

selected_features = ['genres' , 'keywords' , 'tagline' , 'cast' , 'director'] # these 5 things only necessary nothing else
combined_features = ''
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')  
    # conmbinded all the 5 selected features 
    combined_features += movies_data[feature] + ' '

# print (combined_features)




movie_name = input('Enter your favorite movie name : ')


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

similarity = cosine_similarity(feature_vectors)
# print(similarity)

# print(similarity.shape) # (4803,4803)
 
list_of_all_titles = movies_data['title'].tolist()

#print(list_of_all_titles)

find_closed_match = difflib.get_close_matches(movie_name , list_of_all_titles)


print (find_closed_match)
close_movie = find_closed_match[0]


index_of_movie = movies_data[movies_data['title'] == close_movie].index[0]


similarity_score = list(enumerate(similarity[index_of_movie]))
# print(similarity_score)

sorted_simlarity_score = sorted(similarity_score , key = lambda x: x[1], reverse = True)

# print(sorted_simlarity_score)

print("Movie Suggested For You")

for i in range(ask:=int(input("How much movie suggestion you want ?"))):
    print(i+1 ,") " ,  movies_data.iloc[sorted_simlarity_score[i][0]]['title'])
    

    