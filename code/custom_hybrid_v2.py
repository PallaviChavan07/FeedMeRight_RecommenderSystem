import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from code.custom_evaluator import ModelEvaluator
from code.custom_svd import CFRecommender
from code.custom_contentbased import ContentBasedRecommender

start_time = time.time()
#Constants
TEST_USER_ID = 9259
MIN_USERS_INTERACTIONS = 10
MAX_USERS_INTERACTIONS = 20
CB_WEIGHT = 0.3
CF_WEIGHT = 0.7
CB_SCORE_RATING_FACTOR = 5.0
pd.set_option("display.max_rows", None, "display.max_columns", None)

#data
recipe_df = pd.read_csv('../data/clean/recipes.csv')
rating_df = pd.read_csv('../data/clean/ratings.csv')
user_df = pd.read_csv('../data/clean/users.csv')

user_df = user_df.head(1000)
# valid_users_interaction_df is a subset of rating_df
valid_users_interaction_df = pd.merge(rating_df, user_df, on='user_id', how='inner')
merged_df = pd.merge(recipe_df, valid_users_interaction_df, on='recipe_id', how='inner')
# get unique recipes from merged df
unique_valid_recipes = merged_df.recipe_id.unique()
recipe_df = recipe_df[recipe_df['recipe_id'].isin(unique_valid_recipes)]
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]

interactions_train_df, interactions_test_df = train_test_split(interactions_df, test_size=0.20)
print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')
print("--- Total data execution time is %s min ---" %((time.time() - start_time)/60))

#create instance for model evaluator to be used in respective recommenders
model_evaluator = ModelEvaluator(recipe_df, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df)

#Content based
print('\nEvaluating Content-Based Filtering model...')
content_based_recommender_model = ContentBasedRecommender(recipe_df, interactions_full_indexed_df, user_df)
cb_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('Content Based Metrics:\n%s' % cb_metrics)
print("--- Total content based execution time is %s min ---" %((time.time() - start_time)/60))

#collaborative based
print('\nEvaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_recommender_model = CFRecommender(recipe_df, interactions_train_df, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df, user_df)
cf_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('Collaborative SVD Matric Factorization Metrics:\n%s' % cf_metrics)
#print("CF Log: Cols in cf_detailed_results_df", list(cf_detailed_results_df.columns.values))
#print(cf_detailed_results_df.head(5))
print("--- Total Collaborative SVD based execution time is %s min ---" %((time.time() - start_time)/60))

########################################## HYBRID FILTERING BASED ##########################################
class HybridRecommender:
    MODEL_NAME = 'Hybrid'
    def __init__(self, cb_rec_model, cf_rec_model, recipe_df, user_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.recipe_df = recipe_df
        self.user_df = user_df

    def get_model_name(self):
        return self.MODEL_NAME

    def get_recommendation_for_user_calorie_count(self, cal_rec_df, user_id):
        # print("Hybrid: Before calories filter = ", recommendations_df.shape)
        # get calories required for user
        user_calories_per_day = self.user_df.loc[self.user_df['user_id'] == user_id]['calories_per_day'].values
        # print("Hybrid: user calories per day", user_calories_per_day, type(user_calories_per_day), user_calories_per_day[0])
        # divide calories into 1/3rd part
        user_calories = user_calories_per_day[0] / 3
        # consider only those recipes which have calories less than required calories for that user
        cal_rec_df = cal_rec_df[cal_rec_df['calories'] <= user_calories]
        # print("Hybrid: After calories filter = ", recommendations_df.shape)
        return cal_rec_df

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
        #print(recs_df.head(5))

        # Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrength'] = (recs_df['recStrengthCB'] * CB_SCORE_RATING_FACTOR * CB_WEIGHT) + (recs_df['recStrengthCF'] * CF_WEIGHT)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrength', ascending=False).head(topn)
        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions', 'calories']]
        recommendations_df = self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)
        return recommendations_df

print('\nEvaluating Hybrid model...')
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, recipe_df, user_df)
hybrid_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('Hybrid Metrics:\n%s' % hybrid_metrics)

#plot graph
global_metrics_df = pd.DataFrame([cb_metrics, cf_metrics, hybrid_metrics]).set_index('model')
#print(global_metrics_df)
ax = global_metrics_df.transpose().plot(kind='bar', color=['red', 'green', 'blue'])
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()
print("--- Total Hybrid based model execution time is %s min ---" %((time.time() - start_time)/60))