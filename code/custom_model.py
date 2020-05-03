import pandas as pd
import matplotlib.pyplot as plt
import time
import joblib
import sys
import os
from sklearn.model_selection import train_test_split
from custom_evaluator import ModelEvaluator
from custom_svd import CFRecommender
from custom_contentbased import ContentBasedRecommender
from custom_hybrid import HybridRecommender
from custom_popularity import PopularityRecommender
from datetime import datetime

start_time = time.time()
pd.set_option("display.max_rows", None, "display.max_columns", None)

isEval = False
if len(sys.argv) > 1: isEval = sys.argv[1]

#data
recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
rating_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
user_df = pd.read_csv(os.path.realpath('../data/clean/users.csv'))

#user_df = user_df.head(100)
# valid_users_interaction_df is a subset of rating_df
valid_users_interaction_df = pd.merge(rating_df, user_df, on='user_id', how='inner')
merged_df = pd.merge(recipe_df, valid_users_interaction_df, on='recipe_id', how='inner')
# get unique recipes from merged df
unique_valid_recipes = merged_df.recipe_id.unique()
recipe_df = recipe_df[recipe_df['recipe_id'].isin(unique_valid_recipes)]
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]

interactions_train_df, interactions_test_df = train_test_split(interactions_df, test_size=0.20)
#print('# interactions on Train set: %d' % len(interactions_train_df))
#print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')
#print("--- Total data execution time is %s min ---" %((time.time() - start_time)/60))

#create instance for model evaluator to be used in respective recommenders
model_evaluator = ModelEvaluator(recipe_df, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df)

def save_reco_model(filename, model):
    pathtosave = '../models/' + filename + '.mdl'
    joblib.dump(model, os.path.realpath(pathtosave))

def load_reco_model(filename):
    pathtoload = '../models/' + filename + '.mdl'
    return joblib.load(os.path.realpath(pathtoload))

#Content based
if isEval:
    print('\nEvaluating Content-Based...')
    cb_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(load_reco_model('contentbasedmodel'))
    print('Content Based Metrics:\n%s' % cb_metrics)
else:
    print('\nCreating Content-Based Filtering model...')
    content_based_recommender_model = ContentBasedRecommender(recipe_df, interactions_full_indexed_df, user_df)
    save_reco_model('contentbasedmodel', content_based_recommender_model)
    print('Saved contentbasedmodel...')
print("--- Total content based execution time is %s min ---" %((time.time() - start_time)/60))

#collaborative based
if isEval:
    print('\nEvaluating Collaborative...')
    cf_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(load_reco_model('collaborativemodel'))
    print('Collaborative SVD Matric Factorization Metrics:\n%s' % cf_metrics)
else:
    print('\nCreating Collaborative Filtering (SVD Matrix Factorization) model...')
    cf_recommender_model = CFRecommender(recipe_df, interactions_train_df, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df, user_df)
    save_reco_model('collaborativemodel', cf_recommender_model)
    print('Saved collaborativemodel...')
print("--- Total Collaborative SVD based execution time is %s min ---" %((time.time() - start_time)/60))

if isEval:
    print('\nEvaluating Hybrid...')
    hybrid_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(load_reco_model('hybridmodel'))
    print('Hybrid Metrics:\n%s' % hybrid_metrics)
else:
    print('\nCreating Hybrid model...')
    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, recipe_df, user_df)
    save_reco_model('hybridmodel', hybrid_recommender_model)
    print('Saved hybridmodel...')
print("--- Total Hybrid based model execution time is %s min ---" %((time.time() - start_time)/60))

#create and save popularity model as well
if not isEval:
    print('\nCreating Popularity model...')
    popularity_model = PopularityRecommender(interactions_df, recipe_df)
    save_reco_model('popularitymodel', popularity_model)
    print('Saved popularitymodel...')
    print("--- Total popularity based model execution time is %s min ---" %((time.time() - start_time)/60))

if isEval:
    # plot graph
    global_metrics_df = pd.DataFrame([cb_metrics, cf_metrics, hybrid_metrics]).set_index('model')
    # print(global_metrics_df)
    ax = global_metrics_df.transpose().plot(kind='bar', color=['red', 'green', 'blue'], figsize=(15, 8))
    for p in ax.patches:
        # ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        ax.annotate("%.2f" % p.get_height(), (p.get_x(), p.get_height()), ha='center', va='center', xytext=(0, 5),
                    textcoords='offset points')
    # plt.show()
    plotfile = datetime.now().strftime('plot_%b-%d-%Y_%H%M.pdf')
    plt.savefig(os.path.realpath('../plots/%s' % plotfile))
sys.argv.clear()