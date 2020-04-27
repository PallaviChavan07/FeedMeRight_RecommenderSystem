import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

import time
start_time = time.time()

#New Code Start
TEST_USER_ID = 9259
MIN_USERS_INTERACTIONS = 5
#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
pd.set_option("display.max_rows", None, "display.max_columns", None)

#New Code Start
recipe_df = pd.read_csv('../data/original/export_rated_recipes_set.csv')
recipe_df = recipe_df.head(20000)
#print(recipe_df.head(5))

train_rating_df = pd.read_csv('../data/original/core-data-train_rating.csv')
train_rating_df = train_rating_df.head(20000)
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
#print(interactions_df.head(5))
users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
#users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= MIN_USERS_INTERACTIONS].reset_index()[['user_id']]
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

print("--- Total Collaborative SVD based execution time is %s min ---" %((time.time() - start_time)/60))