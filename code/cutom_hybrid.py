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

import time
start_time = time.time()

#New Code Start
TEST_USER_ID = 9259
MIN_USERS_INTERACTIONS = 10
pd.set_option("display.max_rows", None, "display.max_columns", None)

recipe_df = pd.read_csv('../data/original/export_rated_recipes_set.csv')
#print(recipe_df.head(5))
#recipe_df = recipe_df.head(10000)

train_rating_df = pd.read_csv('../data/original/core-data-train_rating.csv')
#train_rating_df = train_rating_df.head(10000)
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')

interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
#interactions_df = interactions_df.set_index('user_id')
#print(interactions_df.head(5))
#print("\nUser Id [", test_user_id, "] details: \n", interactions_df.loc[interactions_df['user_id'] == test_user_id], "\n")

users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
#users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= MIN_USERS_INTERACTIONS].reset_index()[['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, how = 'right', left_on = 'user_id', right_on = 'user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'recipe_id'])['rating'].sum().reset_index()
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
print('# of unique user/item interactions: %d' % len(interactions_full_indexed_df))

########################################## CONTENT BASED ##########################################
#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.003, max_df=0.80, stop_words=stopwords_list)

item_ids = recipe_df['recipe_id'].tolist()
tfidf_matrix = vectorizer.fit_transform(recipe_df['recipe_name'] + "" + recipe_df['ingredients'])
tfidf_feature_names = vectorizer.get_feature_names()

def get_recipes_interacted(user_id):
    # Get the user's data and merge in the information.
    try:
        interacted_items = interactions_full_indexed_df.loc[user_id]
    except:
        interacted_items = None
    return interacted_items

def cb_evaluate_model_for_user(user_id, users_cb_recs_df, k=5):
    if users_cb_recs_df is None:
        return {'p_recall': 0, 'a_recall': 0, 'user_top_k_recos_count': 0, 'user_interated_relevant_count': 0, 'k': k}

    #get top k recos for the user from the complete users_cb_recs_df
    user_top_k_recos = users_cb_recs_df.head(k)

    #get recipes already interacted by user
    user_interact_recipes_df = get_recipes_interacted(user_id)
    #print("user_interact_recipes_df: ", len(user_interact_recipes_df), " for user_id ", user_id)

    #filter out recipes with rating > 3.5 which is our threshold for good vs bad recipes
    user_interated_relevant_df = user_interact_recipes_df.loc[user_interact_recipes_df['rating'] >= 3.5]
    #print("user_interated_relevant_df: ", len(user_interated_relevant_df))

    #merge top k recommended recipes with filtered user interacted recipes to get relevant recommended
    relevant_and_reco_items_df = user_top_k_recos.merge(user_interated_relevant_df, how='inner', on='recipe_id')
    #print("relevant_and_reco_items_df:\n", relevant_and_reco_items_df)

    user_top_k_recos_count = len(user_top_k_recos)
    p_recall = len(relevant_and_reco_items_df) / user_top_k_recos_count if user_top_k_recos_count != 0 else 1
    #print("Pallavi dumb recall", p_recall)

    # Recall@K: Proportion of relevant items that are recommended
    n_rel_and_rec_k = len(relevant_and_reco_items_df)
    n_rel = len(user_interated_relevant_df)
    a_recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    #print("amod yet to correct but dumb recall", a_recall)

    person_metrics = {'p_recall': p_recall,
                      'a_recall': a_recall,
                      'user_top_k_recos_count': user_top_k_recos_count,
                      'user_interated_relevant_count': n_rel,
                      'k': k}

    #print(person_metrics)
    return person_metrics

def cb_recommendatons(cb_objname):
    users_metrics = []
    for idx, user_id in enumerate(list(interactions_full_indexed_df.index.unique().values)):
        users_recs_df = cb_objname.recommend_items(user_id, items_to_ignore=[], topn=10000000000)
        # print(user_id, " Size of users_recs_df: ", users_recs_df.shape)
        singleuser_metric = cb_evaluate_model_for_user(user_id, users_recs_df, k=5)
        users_metrics.append(singleuser_metric)
    print('%d users processed' % idx)
    print('users_metrics: ', len(users_metrics))

    p_detailed_results_df = pd.DataFrame(users_metrics).sort_values('user_top_k_recos_count', ascending=False)
    p_global_recall = p_detailed_results_df['p_recall'].sum() / len(p_detailed_results_df['p_recall'])

    a_detailed_results_df = pd.DataFrame(users_metrics).sort_values('user_interated_relevant_count', ascending=False)
    a_global_recall = a_detailed_results_df['a_recall'].sum() / len(a_detailed_results_df['a_recall'])

    global_metrics = {'modelName': cb_objname.get_model_name(), 'p_global_recall': p_global_recall,
                      'a_global_recall': a_global_recall}

    return global_metrics

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(user_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[user_id]
    try:
        user_interactions_items = interactions_person_df['recipe_id']
    except:
        user_interactions_items = None

    #some users might not have any recipe_id so check for the type
    if type(user_interactions_items) == pd.Series:
        user_item_profiles = get_item_profiles(interactions_person_df['recipe_id'])
        user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    else:
        user_profile_norm = None

    return user_profile_norm

def build_users_profiles():
    interactions_indexed_df = interactions_full_indexed_df[interactions_full_indexed_df['recipe_id'].isin(recipe_df['recipe_id'])]
    user_profiles = {}
    for user_id in interactions_indexed_df.index.unique():
        user_profiles[user_id] = build_users_profile(user_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()
print("\nTotal User Profiles: ", len(user_profiles))
#print(user_profiles)
#myprofile = user_profiles[3324846]
#print(myprofile.shape)
#print(pd.DataFrame(sorted(zip(tfidf_feature_names, user_profiles[3324846].flatten().tolist()), key=lambda x: -x[1])[:20], columns=['token', 'relevance']))
#myprofile = user_profiles[682828]
#print(myprofile.shape)

class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, user_id):
        # Computes the cosine similarity between the user profile and all item profiles
        try:
            cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()
            #print("Take only top ", len(similar_indices), "similar items")
            # Sort the similar items by similarity
            similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        except:
            return None

        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        try:
            similar_items = self._get_similar_items_to_user_profile(user_id)
        except:
            return None
        # early exit
        if similar_items is None: return None

        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        #print("similar_items_filtered \n", similar_items_filtered)
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['recipe_id', 'cb_score']).head(topn)
        if self.items_df is None:
            raise Exception('"items_df" is required in verbose mode')

        recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='recipe_id', right_on='recipe_id')[
            ['cb_score', 'recipe_id']]
        #recommendations_df['user_id'] = user_id

        #print(recommendations_df.shape)
        return recommendations_df

content_based_recommender_model = ContentBasedRecommender(recipe_df)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print('\nEvaluating Content-Based Filtering model...')
cb_metrics = cb_recommendatons(content_based_recommender_model)
print('Global metrics:\n%s' % cb_metrics)
#print(users_cb_recs_df.head(5))

print("--- Total content based execution time is %s min ---" %((time.time() - start_time)/60))
