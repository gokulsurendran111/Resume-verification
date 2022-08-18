#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer - This is used to convert text data into numerical values
from sklearn.metrics.pairwise import cosine_similarity
import os


# In[2]:


def MovieRecommend(movies_data,movie_name,selected_features): 
    combined_features=''
    for i in (selected_features):
        combined_features+=movies_data[i] + ' '
        
    
   # combined_features = movies_data[selected_features[]]+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    ### cosine similarity matrix
    similarity = cosine_similarity(feature_vectors)
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
   # print(find_close_match)
    if len(find_close_match)==0:
        print('Movie not found in the dataset')
    else:
            
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
        print('Movies suggested for you : \n')
        i = 1
        for movie in sorted_similar_movies:
          index = movie[0]
          title_from_index = movies_data[movies_data.index==index]['title'].values[0]
          if (i<30):
            print(i, '.',title_from_index)
            i+=1
          else:
            break


# In[3]:


movies_data =pd.read_csv('movies.csv')
movies_data.head()
movies_data.shape


# In[4]:


movies_data.columns


# In[5]:


## features to look up for the keyword for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[6]:


display (movies_data.isna().sum())


# In[7]:


display (movies_data[selected_features].head())


# In[8]:


display (movies_data[selected_features].isna().sum())


# In[9]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
display (movies_data.head())


# In[11]:


movie_name=input("Enter a movie name or keyword(eg. Iron Man, Avatar, Avengers, ):")
MovieRecommend(movies_data,movie_name,selected_features)


# In[ ]:




