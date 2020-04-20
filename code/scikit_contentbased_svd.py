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


# In[154]:



#recipe_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/export_rated_recipes_set.csv')
#train_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-train_rating.csv')
#test_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/recipe/core-data-test_rating.csv')

recipe_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/original/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/original/core-data-train_rating.csv')
test_rating_df = pd.read_csv('C:/Users/jpall/Anaconda3/envs/conda_venv/conda-meta/data/original/core-data-test_rating.csv')

# movies = pd.read_csv('movielens_data/movies.csv')
# ratings = pd.read_csv('movielens_data/ratings.csv')
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
interactions_df.head(5)


# In[155]:


users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))


# In[156]:


print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


# In[157]:


interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'recipe_id'])
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)


# In[158]:


#interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
 #                                  test_size=0.20,
  #                                 random_state=42)

#interactions_full_df = interactions_full_df[['user_id', 'recipe_id', 'rating']]
#interactions_full_df.columns = ['n_users', 'n_items', 'rating']
df = merged_df[['user_id', 'recipe_id', 'rating']]
#df.columns = ['n_users', 'n_items', 'rating']
interactions_full_df = df
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, test_size=.20)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))


# In[159]:


#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')


# In[160]:


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc['user_id']['recipe_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# In[161]:


#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)
    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index
    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics)                             .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()    


# In[162]:


#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = recipe_df['recipe_id'].tolist()
tfidf_matrix = vectorizer.fit_transform(recipe_df['ingredients'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix


# In[163]:


def get_item_profile(item_id):
    print("#### inside get_item_profile ==>" )
    print("item_id =", item_id )
    #print("item_ids ", item_ids)
    #print("item_ids.index() ", item_ids.index())
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    print("#### inside get_item_profilesSSSS ==>" )
    print("ids =", ids )
    #item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles_list = [get_item_profile(ids)]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles


# In[164]:


def build_users_profile(user_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[user_id]
    print("For user id = ", user_id, "interactions_person_df = ")
    interactions_person_df.head(5)
    user_item_profiles = get_item_profiles(interactions_person_df['recipe_id'])
    user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm


# In[165]:


def build_users_profiles(): 
    interactions_indexed_df = interactions_train_df[interactions_train_df['recipe_id'].isin(recipe_df['recipe_id'])].set_index('user_id')
    print("interactions_indexed_df == ")
    interactions_indexed_df.head()
    user_profiles = {}
    #for user_id in interactions_indexed_df.index.unique():
     #   user_profiles[user_id] = build_users_profile(user_id, interactions_indexed_df)
    #return user_profiles


# In[167]:


user_profiles = build_users_profiles()
#len(user_profiles)


# In[177]:


interactions_indexed_df  = interactions_train_df[interactions_train_df['recipe_id'].isin(recipe_df['recipe_id'])].set_index('user_id')
print("interactions_indexed_df == ")
interactions_indexed_df.head()
user_profiles = {}
for user_id in interactions_indexed_df.index.unique():
        user_profiles[user_id] = build_users_profile(user_id, interactions_indexed_df)
len(user_profiles)


# In[ ]:





# In[ ]:




