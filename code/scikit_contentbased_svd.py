#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
#from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[55]:



#recipe_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/export_rated_recipes_set.csv')
#train_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-train_rating.csv')
#test_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-test_rating.csv')

recipe_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/small_10k/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/small_10k/core-data-train_rating.csv')
test_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/small_10k/core-data-test_rating.csv')

# movies = pd.read_csv('movielens_data/movies.csv')
# ratings = pd.read_csv('movielens_data/ratings.csv')
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
interactions_df.head(5)


# In[56]:


users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))


# In[57]:


print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


# In[62]:


interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'recipe_id'])
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)


# In[63]:


#interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
 #                                  test_size=0.20,
  #                                 random_state=42)

#interactions_full_df = interactions_full_df[['user_id', 'recipe_id', 'rating']]
#interactions_full_df.columns = ['n_users', 'n_items', 'rating']
trainSet, testSet = train_test_split(interactions_full_df,interactions_full_df['user_id'], test_size=.20)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))


# In[ ]:





# In[ ]:




