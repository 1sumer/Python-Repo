#!/usr/bin/env python
# coding: utf-8

# # Indian Regional Recommendation Systems

# The goal for a recommendation system is to extract information from data about the relationship existing between users and products. One of the common usage is to take the products that the user already likes and and answer the question "What other products can be recommended to the user?"
# There are multiple ways of answering the above question. These are the three main types algorithms being used for recommendation systems, to try answering different aspects of recommendations:
# 

# **•	Content Based Filtering**
# 
# 	Used to find products with "similar" attributes. Example: if a person likes movies from action genere then recommend other books from action genere.
# 
# 	The movie will have different attributes, such as, direction, cast, cinematography, story etc. A field specialist will rate all these attributes on a fixed scale.
# 
# 	Based on similarity between these attributes, we can recommend movies to a new user once they have picked up a few movies that they liked.
# 
# 	One major drawback of this method is getting a field expert to rate movies on their attributes. It is time conuming and inefficient.
# 

# **•	Collaberative Filtering**
# 
# 	This is somewhat indirect way of recommending, but is the most used method in the industry. Here we find products liked by "similar" users (they have the same interest as the active user) and recommend them to the active user.
# 
# 	For an active user, who has rated some movies highly in the system; we find other users who have rated these movies in a similar manner (they might have rated other movies as well). These users have commen interests, so we can use other user's ratings as a guide for the likeliness of movie recommendation to the active user.
# 
# 	Here the assumption is that the movies that "similar" users like are similar to each other. In this method also we are measuring similarity of products, however here the similarity is measured indirectly through similarity of users.
# 

# **•	Association Rules Learning**
# 
# 	Here we recommend "complimentary" products, for example, if someone is buying a smartphone then recommend a back cover or a tempered glass etc.
# 
# 	In movie recommendation, the products are substitutable to each other. However, in the above example of smartphone, the products are complimentary.
# 
# 	This algorithm tries to answer the question, "If a person likes to buy smartphones, along with it, what else would they like to buy?" i.e "Which products are associated with each other?"
# 

# **Get movies data from file into DataFrame**

# In[2]:


# Import some basic libraries in python for data preprocessing

import json
import numpy as np
import pandas as pd
# Read movies.json file and load into moviesData

moviesData = []

with open(r"C:\\Users\\syed mubarak pasha\\OneDrive\\Documents\\Python\\Assignment\\Learn\\Indian movie\\movies.json") as moviesFile:
    for line in moviesFile:
        moviesData.append(json.loads(line))

# Now take the moviesData and create a DataFrame

moviesDF = pd.DataFrame.from_records(moviesData)

# View the head of the moviesDF DataFrame

moviesDF.head()


# **Building a content based recommender system**

# Here we will attempt to built two types of content based recommendation systems
# **•	Plot Based Recommendation System:** This model takes movie descriptions and taglines into consideration and provides recommendation with similar plot descriptions.
# 
# **•	Metadata Based Recommendation System:** This model takes different features such as, genres, keywords, cast, and crew etc into consideration and provides recommendations that are most similar.
# 

# >>**I. Plot Based Recommendation System**

# **Goal:** Compute the similarity matrix of all the movies with each other, based on their plot text (using pairwise cosine similarity method)
# 
# **Approach:** Represent plot text (or documents) as vectors i.e a series of numbers, where each number/dimension represents occurance of a specific word in the vocabulary. The size of the vocabulary vector is the number of unique words present when all the documents are put together.
# 
# **Methods:** To create these vectors for all the movies, we use vectorizers;
# 
# •	Count Vectorizer
# 
# •	TF-IDF Vectorizer
# 
# Count Vectorizer
# 
# This is the simplest type of vectorizer.
# 
# Consider an example, where we have three documents (texts)
# 
# A: The sun is a star
# 
# B: My love is like a red, red rose
# 
# C: Mary had a little lamb
# 
# Now our goal is to convert these documents into vectors.
# 
# **Step-1**
# We first compute the vocabulary; i.e the vector comprising of all the unique words present across all the documents.
# 
# V = (the, sun, is, a, star, my, love, like, red, rose, mary, had, little, lamb)
# 
# The size of this vocabulary is 14.
# 
# **Special**
# A common practice is to remove common words e.g. a, the, is, had, my etc.. These are also called 'Stop Words'.
# 
# After removing these stop words;
# 
# V: (like, little, lamb, love, mary, red, rose, sun, star)
# 
# **Step-2**
# Now each document is interpreted as a vector of size 9, where each dimension represents the number of times each word occurs.
# 
# So, using the CountVectorizer approach, A, B and C will be represented as:
# 
# A: (0, 0, 0, 0, 0, 0, 0, 1, 1)
# 
# B: (1, 0, 0, 1, 0, 2, 1, 0, 0)
# 
# C: (0, 1, 1, 0, 1, 0, 0, 0, 0)
# 
# TF-IDF Vectorizer
# 
# Not all words in a document carry equal weight, (for example we saw that stop words have no weightage at all).
# 
# **TF-IDF: Term Frequency - Inverse Document Frequency**
# It assigns the weights to each word according to the formula (for every word i in document j):
# 
# $$wi,j=tfi,j×log(Ndfi)wi,j=tfi,j×log(Ndfi)$$
# 
# Where,
# 
# wi,jwi,j is the weight of word i in document j
# 
# tfi,jtfi,j is the term frequency for word i in document j
# 
# NN is the total number of documents
# 
# dfidfi is the number of documents that contain the term i
# 
# **pro:** Speeds up the computation of cossine similarity score
# 
# Cosine Similarity:
# 
# Cosine similarity between two documents, x and y,
# 
# $$cosine(x,y)=x.yT||x||.||y||cosine(x,y)=x.yT||x||.||y||$$
# 
# It takes a value between -1 and 1, higher the score, more similar the documents are to each other.
# 
# Steps for building a Plot based Recommender
# 
# It takes a movie title as an argument and outputs a list of movies that are most similar based on their plots.
# 
# **Steps**
# 1.	Clean the data to the format required to build the model
# 
# 2.	Create TF-IDF vectors for the plot description of every movie
# 
# 3.	Compute the pairwise cosine similarity between every movie
# 
# 4.	Write a recommender function that takes in a movie title as an argument and outputs movies most similar to it based on the plot.
# 

# **1. Prep the data**

# In[3]:


# Let's create a dataframe with name and description
plotDF = moviesDF.loc[:, ['name', 'description']]

# Convert both columns to lowercase
plotDF.loc[:, 'name'] = plotDF.loc[:, 'name'].apply(lambda x: x.lower())
plotDF.loc[:, 'description'] = plotDF.loc[:, 'description'].apply(lambda x: x.lower())

# Drop all the rows which are empty
plotDF = plotDF[plotDF['description'] != '']

# Now drop duplicates in the plotDF
plotDF = plotDF.drop_duplicates()

plotDF.head()


# In[4]:


# Lets quickly view the info of plotDF
plotDF.info()


# **2. Create the TF-IDF matrix**
# 
# Each row of this matrix represents the TF_IDF vector of the description feature of the corresponding movie in the plotDF data frame.

# In[5]:


# Import the TfIdfVectorizer from scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF IDF Vectorizer Object
# with the removal of english stopwords turned on
tfidf = TfidfVectorizer(stop_words = 'english')

# Now costruct the TF-IDF Matrix by applying the fit_transform method on the description feature
tfidf_matrix = tfidf.fit_transform(plotDF['description'])

# View the shape of the TfIdf_matrix
tfidf_matrix.shape


# **3. Computing the cosine similarity score**
# 
# Here we are going to create a matrix of size 1751 x 1751, where i-th row and j-th column column represents the similarity score between movies i and j.
# 
# This matrix will be symmetric in nature and all the elements along the diagonal will be 1, since it the similarity score of the matrix with itself.
# 
# Also, as we have represented the movie plots as TF-IDF vectors, they have the magnitude of 1. So, we need not calculate the magnitude of the dot product (denominator of the cosine similarity function will always be 1).
# 
# So out cosine similarity function reduces to:
# 
# $$cosine(x,y)=x.yTcosine(x,y)=x.yT$$
# 

# In[6]:


# Import linear_kernel from scikit-learn to compute the dot product
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix by using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Let's view the shape of the cosine similarity matrix
print("Shape of cosine similarity matrix: ", cosine_sim.shape)

# Let's quickly view how the matrix looks like in first few column and rows
cosine_sim[0:5, 0:5]


# In[7]:


# Construct a pandas series of movie name as index and index as value
indices = pd.Series(plotDF.index, index = plotDF['name'])
indices.head()


# **4. Build the recommender function**
# 
# **Steps**
# 
# 1.	Input the title of the movie as an argument
# 
# 2.	Obtain the index of the movie from the indeces series
# 
# 3.	Get the list of cosine similarity scores for that particular movie with all movies using cosine_sim matrix.And, convert this in to a list of tuples where the first element is the position and second element is the similarity score.
# 
# 4.	Sort this list of tuples on the basis of the cosine similarity scores.
# 
# 5.	Get the top 10 elements of the list, while ignoring the first element as it refers to the similarity score with itself.
# 
# 6.	Return the titles corresponding to the indices of the top 10 elements.
# 

# In[9]:


# Function for returning top 10 movies for a movie title as input
def plot_based_recommender(title, df = plotDF, cosine_sim = cosine_sim, indices = indices):
  # Convert title to lower-case
  title = title.lower()

  # Obtain the index of the movie that matched the title 
  try:
    idx = indices[title]
  except KeyError:
    print('Movie does not exist :(')
    return False

  # Get the pairwise similarity score of all the movies with that movie
  # and convert it into a list of tuples (position, similarity score)
  sim_scores = list(enumerate(cosine_sim[idx]))

  # Sort the movies based on the cosine similarity scores
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

  # Get the scores of the top 10 most similar movies. Ignore the first movie.
  sim_scores = sim_scores[1:11]

  # get the movie indices
  movie_indices = [sim_score[0] for sim_score in sim_scores]

  # Return the top 10 similar movies
  return df['name'].iloc[movie_indices]


# In[10]:


plot_based_recommender('Pyaar Ka Punchnama')


# >>**II. Metadata Based Recommender System**

# To build this model, we will be using the following meta-data:
# 
# •	genre
# 
# •	language
# 
# •	director
# 
# •	cast
# 
# •	smovie description
# 
# Apart from the difference in data being used, we will largely follow the same steps as the plot based recommender system.
# 

# **Prep the Data**

# In[30]:


# Prepare the data
metaDF = moviesDF[['name', 'genre', 'language', 'director', 'cast', 'description']]
metaDF.head()


# In[12]:


# We want to keep only the first 3 genre and cast (actors) in the list format
# We want to keep only the first director

metaDF.loc[:, 'genre'] = metaDF.loc[:, 'genre'].apply(lambda x: x[:3] if len(x) > 3 else x)
metaDF.loc[:, 'cast'] = metaDF.loc[:, 'cast'].apply(lambda x: x[:3] if len(x) > 3 else x)
metaDF.loc[:, 'director'] = metaDF.loc[:, 'director'].apply(lambda x: x[:1])
metaDF.head()


# In[13]:


def sanitize(x):
    if isinstance(x, list):
        # Strip spaces
        return [i.replace(" ", "") for i in x]
    else:
        # if it is empty, return an empty string
        if isinstance(x, str):
            return x.replace(" ", "")
        else: 
            return ''


# In[14]:


#Apply the generate_list function to cast, keywords, director and genres 
for feature in ['director', 'cast']:
    metaDF[feature] = metaDF[feature].apply(sanitize)


# In[15]:


metaDF.head()


# The spaces between the names of actors and directors can create problems as they can be considered as separate words. We do not want that, so our next step is to strip the spaces between the names.

# In[18]:


# Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['genre']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['cast'] + [x['name']] + [x['description']])

# Create the new soup feature 
metaDF['soup'] = metaDF.apply(create_soup, axis=1)

#Display the soup of the first movie 
metaDF.iloc[0]['soup']


# In[19]:


# Let's create a dataframe with name and soup
soupDF = metaDF.loc[:, ['name', 'soup']]

# Convert both columns to lowercase
soupDF.loc[:, 'name'] = soupDF.loc[:, 'name'].apply(lambda x: x.lower())
soupDF.loc[:, 'soup'] = soupDF.loc[:, 'soup'].apply(lambda x: x.lower())

# Drop all the rows which are empty
soupDF = soupDF[soupDF['soup'] != '']

# Now drop duplicates in the soupDF
soupDF = soupDF.drop_duplicates()

soupDF.head()
print(soupDF.shape)


# In[20]:


# Import the TfIdfVectorizer from scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF IDF Vectorizer Object
# with the removal of english stopwords turned on
tfidf = TfidfVectorizer(stop_words = 'english')

# Now costruct the TF-IDF Matrix by applying the fit_transform method on the description feature
tfidf_matrix = tfidf.fit_transform(soupDF['soup'])

# View the shape of the TfIdf_matrix
tfidf_matrix.shape


# In[21]:


# Import linear_kernel from scikit-learn to compute the dot product
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix by using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Let's view the shape of the cosine similarity matrix
print("Shape of cosine similarity matrix: ", cosine_sim.shape)

# Let's quickly view how the matrix looks like in first few column and rows
cosine_sim[0:5, 0:5]


# In[22]:


# Construct a pandas series of movie name as index and index as value
indices = pd.Series(soupDF.index, index = soupDF['name'])
indices.head()


# In[24]:


# Function for returning top 10 movies for a movie title as input
def plot_based_recommender(title, df = soupDF, cosine_sim = cosine_sim, indices = indices):
  # Convert title to lower-case
  title = title.lower()

  # Obtain the index of the movie that matched the title 
  try:
    idx = indices[title]
  except KeyError:
    print('Movie does not exist :(')
    return False

  # Get the pairwise similarity score of all the movies with that movie
  # and convert it into a list of tuples (position, similarity score)
  sim_scores = list(enumerate(cosine_sim[idx]))

  # Sort the movies based on the cosine similarity scores
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

  # Get the scores of the top 10 most similar movies. Ignore the first movie.
  sim_scores = sim_scores[1:11]

  # get the movie indices
  movie_indices = [sim_score[0] for sim_score in sim_scores]

  # Return the top 10 similar movies
  return df['name'].iloc[movie_indices]


# In[25]:


plot_based_recommender('Pyaar Ka Punchnama')


# In[26]:


# database 
db = {} 
db['cosine_sim'] = cosine_sim
db['indices'] = indices

import pickle

dbfile = open('examplePickle', 'ab') 
pickle.dump(db, dbfile)                      
dbfile.close() 


# In[ ]:




