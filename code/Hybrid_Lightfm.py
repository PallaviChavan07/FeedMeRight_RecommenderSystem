## Source -- https://github.com/youonf/recommendation_system/blob/master/Hybrid%20system/Hybrid%20with%20Lightfm.ipynb

import numpy as np
import scipy
import pandas as pd
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import sys
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder(sparse=True)

recipe_df = pd.read_csv('../data/original/export_rated_recipes_set.csv')
rating_df = pd.read_csv('../data/original/core-data-train_rating.csv')
#recipe_df = recipe_df.head(15000)
#rating_df = rating_df.head(15000)
print(recipe_df.columns.values)
short_recipe_df = recipe_df[['recipe_id', 'recipe_name',  'ingredients']]
#print("short_recipe_df = ", short_recipe_df.head())
#recipe_df.columns = ['recipe_id', 'recipe_name',  'ingredients', axis=1]
merged_df = pd.merge(right=short_recipe_df, left=rating_df,how='inner', on='recipe_id')
df_item = merged_df[['recipe_id','ingredients']]
merged_df.sort_values('user_id',inplace=True)
df_item.dropna(inplace=True)
merged_df.dropna(inplace=True)


#created a sparse matrix of item feature to fit in LightFM model
item_features=encode.fit_transform(merged_df[['recipe_id','ingredients']])
#print(item_features)

#Fit Lightfm hybrid model with author
from lightfm.data import Dataset
dataset=Dataset()
dataset.fit(merged_df.user_id.values,merged_df.recipe_id.values,item_features = df_item['ingredients'].values)
# fit ratings, book isbn and book features to the model

item_sub = merged_df[['recipe_id', 'ingredients']]
item_tuples = [tuple(x) for x in item_sub.values]

user_sub = merged_df[['user_id', 'recipe_id']]
user_tuples = [tuple(x) for x in user_sub.values]
(interactions, weights) = dataset.build_interactions(user_tuples)
interactions
# build interaction on what item the user rated and the cooresponing item feature

from lightfm.cross_validation import random_train_test_split
train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(seed=111))


from lightfm import LightFM
# Define a new model instance
model = LightFM(loss='warp',
                no_components=20)

# Fit the hybrid model, remember to pass in item features.
model = model.fit(train,
                item_features=item_features,
                epochs=10,
                num_threads=4)


from lightfm.evaluation import auc_score
# Don't forget the pass in the item features again!
train_auc = auc_score(model,
                      train,
                      item_features=item_features).mean()
print('Hybrid training set AUC: %s' % train_auc)

test_auc = auc_score(model,
                    test,
                    train_interactions=train,
                    item_features=item_features).mean()
print('Hybrid test set AUC: %s' % test_auc)
#