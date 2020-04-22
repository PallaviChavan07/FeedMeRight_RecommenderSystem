#https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101#Collaborative-Filtering-model
import matplotlib
import numpy as np
import scipy
import pandas as pd
import math
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

articles_df = pd.read_csv('../data/original/articles/shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)

interactions_df = pd.read_csv('../data/original/articles/users_interactions.csv')
interactions_df.head(10)
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0,
   'BOOKMARK': 2.5,
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,
}
interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, how = 'right', left_on = 'personId', right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


def smooth_user_preference(x):
    return math.log(1 + x, 2)

interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength'].sum().apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, stratify=interactions_full_df['personId'], test_size=0.20, random_state=42)
print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')

def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# Top-N accuracy metrics consts
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
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['contentId'].values
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
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(), 'recall@5': global_recall_at_5, 'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df

model_evaluator = ModelEvaluator()

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)].sort_values('eventStrength', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='contentId', right_on='contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df

popularity_model = PopularityRecommender(item_popularity_df, articles_df)
print('\nEvaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('Global metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)


#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.003, max_df=0.5, max_features=5000, stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
#print(tfidf_matrix)


def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile


def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles


def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])

    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
    # Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
        user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm


def build_users_profiles():
    interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
        .isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()
print("\nTotal User Profiles: ", len(user_profiles))
myprofile = user_profiles[-1479311724257856983]
#print(myprofile.shape)
pd.DataFrame(sorted(zip(tfidf_feature_names, user_profiles[-1479311724257856983].flatten().tolist()), key=lambda x: -x[1])[:20], columns=['token', 'relevance'])

class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='contentId', right_on='contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


content_based_recommender_model = ContentBasedRecommender(articles_df)

print('\nEvaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('Global metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', columns='contentId', values='eventStrength').fillna(0)
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
U.shape
Vt.shape
sigma = np.diag(sigma)
sigma.shape
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings

all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)

len(cf_preds_df.columns)


class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='contentId', right_on='contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

print('\nEvaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('Global metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)


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
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, topn=1000).rename(columns={'recStrength': 'recStrengthCB'})

        # Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, topn=1000).rename(columns={'recStrength': 'recStrengthCF'})

        # Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df, how='outer', left_on='contentId', right_on='contentId').fillna(0.0)

        # Computing a hybrid recommendation score based on CF and CB scores
        # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='contentId', right_on='contentId')[['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df, cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)
print('\nEvaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('Global metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)

global_metrics_df = pd.DataFrame([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics]).set_index('modelName')
print(global_metrics_df)
#%matplotlib inline
ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how = 'left', left_on = 'contentId', right_on = 'contentId') \
                          .sort_values('eventStrength', ascending = False)[['eventStrength', 'contentId', 'title', 'url', 'lang']]
inspect_interactions(-1479311724257856983, test_set=False).head(20)
hybridmodelrecoSingleUserdf = hybrid_recommender_model.recommend_items(-1479311724257856983, topn=20, verbose=True)
print(hybridmodelrecoSingleUserdf)



#Output:
# # users: 1895
# # users with at least 5 interactions: 1140
# # of interactions: 72312
# # of interactions from users with at least 5 interactions: 69868
# # of unique user/item interactions: 39106
# # interactions on Train set: 31284
# # interactions on Test set: 7822
#
# Evaluating Popularity recommendation model...
# 1139 users processed
# Global metrics:
# {'modelName': 'Popularity', 'recall@5': 0.2418818716440808, 'recall@10': 0.3725389925850166}
#
# Total User Profiles:  1140
#
# Evaluating Content-Based Filtering model...
# 1139 users processed
# Global metrics:
# {'modelName': 'Content-Based', 'recall@5': 0.10163641012528765, 'recall@10': 0.17220659677831757}
#
# Evaluating Collaborative Filtering (SVD Matrix Factorization) model...
# 1139 users processed
# Global metrics:
# {'modelName': 'Collaborative Filtering', 'recall@5': 0.33392994119151115, 'recall@10': 0.46803886474047557}
#
# Evaluating Hybrid model...
# 1139 users processed
# Global metrics:
# {'modelName': 'Hybrid', 'recall@5': 0.333802096650473, 'recall@10': 0.465993352083866}
#                          recall@5  recall@10
# modelName
# Content-Based            0.101636   0.172207
# Popularity               0.241882   0.372539
# Collaborative Filtering  0.333930   0.468039
# Hybrid                   0.333802   0.465993
#     recStrengthHybrid  ...  lang
# 0           25.425245  ...    en
# 1           25.369932  ...    en
# 2           24.701694  ...    pt
# 3           24.377750  ...    en
# 4           24.362064  ...    en
# 5           24.183549  ...    en
# 6           24.162866  ...    en
# 7           23.921336  ...    en
# 8           23.864363  ...    en
# 9           23.804789  ...    en
# 10          23.529632  ...    en
# 11          23.313283  ...    pt
# 12          23.189662  ...    en
# 13          22.715206  ...    en
# 14          22.553447  ...    en
# 15          22.442176  ...    en
# 16          22.339452  ...    en
# 17          22.311658  ...    pt
# 18          22.264841  ...    en
# 19          22.231268  ...    en
#
# [20 rows x 5 columns]










########################################## POPULARITY BASED ##########################################
#Computes the most popular items
#item_popularity_df = interactions_full_df.groupby('recipe_id').sum().reset_index()
#item_popularity_df = interactions_full_df.groupby('recipe_id')['rating'].sum().sort_values(ascending=False).reset_index()
#item_popularity_df.head(10)

# class PopularityRecommender:
#     MODEL_NAME = 'Popularity'
#     def __init__(self, popularity_df, items_df=None):
#         self.popularity_df = popularity_df
#         self.items_df = items_df
#
#     def get_model_name(self):
#         return self.MODEL_NAME
#
#     def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
#         # Recommend the more popular items that the user hasn't seen yet. (maybe needs sorting here?)
#         #recommendations_df = self.popularity_df[~self.popularity_df['recipe_id'].isin(items_to_ignore)].head(topn)
#         recommendations_df = self.popularity_df[~self.popularity_df['recipe_id'].isin(items_to_ignore)].sort_values('rating', ascending=False).head(topn)
#
#         if verbose:
#             if self.items_df is None:
#                 raise Exception('"items_df" is required in verbose mode')
#
#             recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recipe_id', 'recipe_name', 'ingredients', 'nutritions']]
#
#         return recommendations_df

#popularity_model = PopularityRecommender(item_popularity_df, recipe_df)
#print('\nEvaluating Popularity recommendation model...')
#pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
#print('Global metrics:\n%s' % pop_global_metrics)
#print(pop_detailed_results_df.head(5))

