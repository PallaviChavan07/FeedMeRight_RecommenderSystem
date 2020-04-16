#!/usr/bin/env python
# coding: utf-8

# In[38]:


## Source - https://blog.dominodatalab.com/recommender-systems-collaborative-filtering/
## Source - https://www.kaggle.com/robottums/hybrid-recommender-systems-with-surprise

## Collaborative filtering based on item item similarity
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNBasic
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import split
from collections import defaultdict


# In[15]:


recipe_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-train_rating.csv')
test_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-test_rating.csv')

df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
trainSet, testSet = train_test_split(data, test_size=.25, random_state=0)

# data = surprise.Dataset.load_from_df(rawTrain,reader)
# holdout = surprise.Dataset.load_from_df(rawholdout,reader)

 # split data into folds.
kSplit = split.KFold(n_splits=10, shuffle=True) 

# compute  similarities between items
sim_options = sim_options = {'name': 'cosine',
               'user_based': False  
               }


collabKNN  = KNNBasic(k=40,sim_options=sim_options)                                                       
rmseKNN = []
rmseSVD = []
rmseCo = []
rmseSlope = []


for trainset, testset in kSplit.split(data): #iterate through the folds.
    collabKNN.fit(trainset)
    predictionsKNN = collabKNN.test(testset)
    rmseKNN.append(accuracy.rmse(predictionsKNN,verbose=True))#get root means squared error
print(rmseKNN) 


# for trainset, testset in data.folds(): 
#     algo.fit(trainset)                             
#     predictions = algo.test(testset)
#     rmse(predictions)


# trainingSet = data.build_full_trainset()
# sim_options = {
#     'name': 'cosine',
#     'user_based': False
# }
 
# knn = KNNBasic(sim_options=sim_options)


# In[34]:


#helper method to get only top 5 recipe recommendation for each user.

from collections import defaultdict
 
def get_top5_recommendations(predictions, topN = 5):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs


# In[35]:


# helper method to create a dictionary that maps each recipe is to it's name
top_5 = get_top5_recommendations(predictionsKNN)

# Print the recommended items for each user
for uid, user_ratings in top_5.items():
    if(uid == 16):
        print(uid, [iid for (iid, _) in user_ratings])


# In[39]:


## Different way to get recommendation for single user
## Don't want to recommend same recipes that they already have tried

# Get a list of all recipe ids
unique_recipe_ids = df['recipe_id'].unique()
# Get a list of all recipe ids that has beenn rated by user 39
recipe_ids_ratedby_user = df.loc[df['user_id'] == 39, 'recipe_id']
# remove the recipe_ids that user -- (here 39) has rated from the list of all recipe ids.
recipeids_to_pred = np.setdiff1d(unique_recipe_ids, recipe_ids_ratedby_user)
print("unique_recipe_ids = ", len(unique_recipe_ids))
print("recipe_ids_ratedby_user = ", len(recipe_ids_ratedby_user))
print("recipeids_to_pred = ",len(recipeids_to_pred ))


# In[42]:


testset = [[39, iid, 1.] for iid in recipeids_to_pred]
predictions = collabKNN.test(testset)
predictions[0]


# In[43]:


pred_ratings = np.array([pred.est for pred in predictions])
# find the index of the maximum predicted rating 
i_max = pred_ratings.argmax()
# use this to find the corresponding recipeid to recommend
recipe_id = recipeids_to_pred[i_max]
print("top recipe for user 39 is with recipe id = ", recipe_id, " with predicted rating = ", pred_ratings[i_max])


# In[1]:




