from sklearn.model_selection import train_test_split
from code.custom_hybrid import HybridRecommender
from code.custom_popularity import PopularityRecommender
from code.custom_svd import CFRecommender
from code.custom_contentbased import ContentBasedRecommender
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import sys
import time
start_time = time.time()

isNewUser = False
newuser_cal_count = 1000
REC_FOR_USER_ID = 0
if len(sys.argv) > 1:
    newuser_cal_count = sys.argv[0]
    isNewUser = sys.argv[1]
else:
    REC_FOR_USER_ID = sys.argv[0]

#data
recipe_df = pd.read_csv('../data/clean/recipes.csv')
rating_df = pd.read_csv('../data/clean/ratings.csv')
user_df = pd.read_csv('../data/clean/users.csv')

#for user id get already rated recipe ids in a list
recipes_to_ignore_list = rating_df.loc[rating_df['user_id'] == REC_FOR_USER_ID]['recipe_id'].values.tolist()
#print("recipes_to_ignore_list: ", recipes_to_ignore_list)

#valid_users_interaction_df is a subset of rating_df
valid_users_interaction_df = pd.merge(rating_df, user_df, on='user_id', how='inner')
merged_df = pd.merge(recipe_df, valid_users_interaction_df, on='recipe_id', how='inner')
#get unique recipes from merged df
unique_valid_recipes = merged_df.recipe_id.unique()
recipe_df = recipe_df[recipe_df['recipe_id'].isin(unique_valid_recipes)]
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]

#split data into train and test
interactions_train_df, interactions_test_df = train_test_split(interactions_df, test_size=0.20)

#Indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')
print("--- Total data execution time is %s min ---" %((time.time() - start_time)/60))

#if the recipes list is empty the user has not rated anything and safe to treat as new user. In this case, only run popularity model.
if len(recipes_to_ignore_list) < 1 or isNewUser:
    popularity_model = PopularityRecommender(interactions_df, recipe_df)
    pop_final_top10_recommendation_df = popularity_model.recommend_items(topn=10, pd=pd, newuser_cal_count=newuser_cal_count)
    if not isNewUser: print("\n Entered user Id does not exists in the system. Showing Recommendations based on popularity model.\n", pop_final_top10_recommendation_df)
    else: print('\nRecommendations based on popularity model are:\n', pop_final_top10_recommendation_df)
    print("--- Total Popularity based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))
else:
    print('\nContent-Based Recommendation using CB model...')
    content_based_recommender_model = ContentBasedRecommender(recipe_df, interactions_full_indexed_df, user_df)
    cb_final_top10_recommendation_df = content_based_recommender_model.recommend_items(REC_FOR_USER_ID, recipes_to_ignore_list, 10)
    print(cb_final_top10_recommendation_df)
    print("--- Total content based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

    print('\nCollaborative Filtering Recommendation using SVD Matrix Factorization...')
    cf_recommender_model = CFRecommender(recipe_df, interactions_train_df, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df, user_df)
    cf_final_top10_recommendation_df = cf_recommender_model.recommend_items(REC_FOR_USER_ID, recipes_to_ignore_list, 10)
    print(cf_final_top10_recommendation_df)
    print("--- Total SVD based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

    print('\nRecommendation using Hybrid model...')
    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, recipe_df, user_df)
    hybrid_final_top10_recommendation_df = hybrid_recommender_model.recommend_items(REC_FOR_USER_ID, recipes_to_ignore_list, 10)
    print(hybrid_final_top10_recommendation_df)
    print("--- Total Hybrid based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))
