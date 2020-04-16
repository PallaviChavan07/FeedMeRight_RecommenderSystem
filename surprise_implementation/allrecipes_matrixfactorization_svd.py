#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ## Source - https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b
# ## Notebook Code Reference - https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Movielens%20Recommender%20Metrics.ipynb
# import nltk
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import csv
# import sys

# from surprise import KNNBasic
# from surprise import Dataset                                                     
# from surprise import Reader                                                      
# from surprise import dump
# from surprise.accuracy import rmse
# from surprise.model_selection import train_test_split
# from surprise import SVD


from surprise import KNNBasic
import pandas as pd
from surprise import dump
from surprise.accuracy import rmse
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict


recipe_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-train_rating.csv')
test_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-test_rating.csv')

# movies = pd.read_csv('movielens_data/movies.csv')
# ratings = pd.read_csv('movielens_data/ratings.csv')
df = pd.merge(recipe_df, rating_df, on='recipe_id', how='inner')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'recipe_name', 'rating']], reader)
trainSet, testSet = train_test_split(data, test_size=.25, random_state=0)
algo = SVD(random_state=0)
algo.fit(trainSet)
predictions = algo.test(testSet)

def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)
def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
    
print("RMSE: ", RMSE(predictions))
print("MAE: ", MAE(predictions))


# In[ ]:




