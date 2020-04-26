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
old_stdout = sys.stdout

filename = datetime.now().strftime('mylogfile_%b-%d-%Y_%H%M.log')
log_file = open("../logs/%s" %filename,"w")
sys.stdout = log_file
# recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
# train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')
# test_rating_df = pd.read_csv('../data/small_10k/core-data-test_rating.csv')
#
# merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
# interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
#
# users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
# print('# users: %d' % len(users_interactions_count_df))
# users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
# print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
# print('# of interactions: %d' % len(interactions_df))
# interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, how = 'right', left_on = 'user_id', right_on = 'user_id')
# print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
#
# interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'recipe_id'])
# print('# of unique user/item interactions: %d' % len(interactions_full_df))
# interactions_full_df.head(10)
#
# interactions_full_df = merged_df[['user_id', 'recipe_id', 'rating']]
# interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, test_size=.20)
# print('# interactions on Train set: %d' % len(interactions_train_df))
# print('# interactions on Test set: %d' % len(interactions_test_df))

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 50

#New Code Start
recipe_df = pd.read_csv('../data/original/export_rated_recipes_set.csv')
recipe_df = recipe_df.head(20000)
#print(recipe_df.head(5))

train_rating_df = pd.read_csv('../data/original/core-data-train_rating.csv')
train_rating_df = train_rating_df.head(20000)
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
print(interactions_df.head(5))
users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
#users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 50].reset_index()[['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, how = 'right', left_on = 'user_id', right_on = 'user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'recipe_id'])['rating'].sum().reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, test_size=0.20)
print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
#New Code End

#Indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')

def get_items_interacted(user_id, interactions_df):
    # Get the user's data and merge in the information.
    try:
        interacted_items = interactions_df.loc[user_id]['recipe_id']
    except:
        interacted_items = None

    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

class ModelEvaluator:
    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted(user_id, interactions_full_indexed_df)
        all_items = set(recipe_df['recipe_id'])
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

    def evaluate_model_for_user(self, model, user_id):
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_testset['recipe_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['recipe_id'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['recipe_id'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(user_id, items_to_ignore=get_items_interacted(user_id, interactions_train_indexed_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            if not person_recs_df is None:
                valid_recs_df = person_recs_df[person_recs_df['recipe_id'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['recipe_id'].values
            else:
                #this way we can still get person_metrics
                valid_recs = None

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}

        #if person_recs_df is None: print(person_metrics)
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, user_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, user_id)
            person_metrics['_user_id'] = user_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(), 'recall@5': global_recall_at_5, 'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df

model_evaluator = ModelEvaluator()

########################################## CONTENT BASED ##########################################
#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.003, max_df=0.5, max_features=5000, stop_words=stopwords_list)

item_ids = recipe_df['recipe_id'].tolist()
tfidf_matrix = vectorizer.fit_transform(recipe_df['recipe_name'] + "" + recipe_df['ingredients'])
tfidf_feature_names = vectorizer.get_feature_names()

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
    interactions_indexed_df = interactions_train_df[interactions_train_df['recipe_id'].isin(recipe_df['recipe_id'])].set_index('user_id')
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

    def _get_similar_items_to_user_profile(self, user_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        try:
            cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()[-topn:]
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
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['recipe_id', 'recStrength']).head(topn)
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions']]

        return recommendations_df


content_based_recommender_model = ContentBasedRecommender(recipe_df)
print('\nEvaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('Global metrics:\n%s' % cb_global_metrics)
#print("CB Log: Cols in cb_detailed_results_df", list(cb_detailed_results_df.columns.values))
#print(cb_detailed_results_df.head(5))

########################################## COLLABORATIVE FILTERING BASED ##########################################

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='user_id', columns='recipe_id', values='rating').fillna(0)
users_items_pivot_matrix_df.head(10)

users_items_pivot_matrix = users_items_pivot_matrix_df
users_items_pivot_matrix[:10]
users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]
users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
#U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
#print(U.shape)
#print(Vt.shape)
sigma = np.diag(sigma)
#print(sigma.shape)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
#print(all_user_predicted_ratings)

all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
#print("CF log:", cf_preds_df.head(5))
#print("CF log:", len(cf_preds_df.columns))

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        try:
            sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        except:
            return None

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['recipe_id'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions']]

        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df, recipe_df)
print('\nEvaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('Global metrics:\n%s' % cf_global_metrics)
#print("CF Log: Cols in cf_detailed_results_df", list(cf_detailed_results_df.columns.values))
#print(cf_detailed_results_df.head(5))

########################################## HYBRID FILTERING BASED ##########################################
class HybridRecommender:
    MODEL_NAME = 'Hybrid'
    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Getting the top-1000 Content-based filtering recommendations
        try:
            cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        except:
            return None

        # Getting the top-1000 Collaborative filtering recommendations
        try:
            cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        except:
            return None

        # Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df, how='outer', left_on='recipe_id', right_on='recipe_id').fillna(0.0)

        # Computing a hybrid recommendation score based on CF and CB scores
        # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions']]

        return recommendations_df

hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, recipe_df, cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)
print('\nEvaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('Global metrics:\n%s' % hybrid_global_metrics)
#print(hybrid_detailed_results_df.head(5))

#plot graph
#global_metrics_df = pd.DataFrame([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics]).set_index('modelName')
global_metrics_df = pd.DataFrame([cb_global_metrics, cf_global_metrics, hybrid_global_metrics]).set_index('modelName')
#print(global_metrics_df)
ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
#plt.show()
plotfile = datetime.now().strftime('plot_%b-%d-%Y_%H%M.pdf')
plt.savefig('../plots/%s' %plotfile)

def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(recipe_df, how = 'left', left_on = 'recipe_id', right_on = 'recipe_id') \
                          .sort_values('rating', ascending = False)[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions']]
#inspect_interactions(3324846, test_set=False).head(5)
#hybridmodelrecoSingleUserdf = hybrid_recommender_model.recommend_items(3324846, topn=5, verbose=True)
#print("\nHybrid Model Show Top 5 Recommendations for user [", 3324846, "]\n", hybridmodelrecoSingleUserdf)






sys.stdout = old_stdout
log_file.close()