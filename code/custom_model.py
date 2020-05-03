import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from code.custom_evaluator import ModelEvaluator
from code.custom_svd import CFRecommender
from code.custom_contentbased import ContentBasedRecommender
from code.custom_hybrid import HybridRecommender
from datetime import datetime

start_time = time.time()
pd.set_option("display.max_rows", None, "display.max_columns", None)

#data
recipe_df = pd.read_csv('../data/clean/recipes.csv')
rating_df = pd.read_csv('../data/clean/ratings.csv')
user_df = pd.read_csv('../data/clean/users.csv')

user_df = user_df.head(100)
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

print('\nEvaluating Hybrid model...')
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, recipe_df, user_df)
hybrid_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('Hybrid Metrics:\n%s' % hybrid_metrics)

#plot graph
global_metrics_df = pd.DataFrame([cb_metrics, cf_metrics, hybrid_metrics]).set_index('model')
#print(global_metrics_df)
ax = global_metrics_df.transpose().plot(kind='bar', color=['red', 'green', 'blue'], figsize=(15,8))
for p in ax.patches:
    #ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    ax.annotate("%.2f" % p.get_height(), (p.get_x(), p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
#plt.show()
plotfile = datetime.now().strftime('plot_%b-%d-%Y_%H%M.pdf')
plt.savefig('../plots/%s' %plotfile)

print("--- Total Hybrid based model execution time is %s min ---" %((time.time() - start_time)/60))