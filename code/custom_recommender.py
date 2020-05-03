import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import sys
import joblib
import time
import os
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
rating_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
#for user id get already rated recipe ids in a list
recipes_to_ignore_list = rating_df.loc[rating_df['user_id'] == REC_FOR_USER_ID]['recipe_id'].values.tolist()
#print("recipes_to_ignore_list: ", recipes_to_ignore_list)

def load_reco_model(filename):
    pathtoload = '../models/' + filename + '.mdl'
    return joblib.load(os.path.realpath(pathtoload))

#if the recipes list is empty the user has not rated anything and safe to treat as new user. In this case, only run popularity model.
if len(recipes_to_ignore_list) < 1 or isNewUser:
    popularity_model = load_reco_model('popularitymodel')
    pop_final_top10_recommendation_df = popularity_model.recommend_items(topn=10, pd=pd, newuser_cal_count=newuser_cal_count)
    if not isNewUser: print("\n Entered user Id does not exists in the system.\nShowing Recommendations based on popularity model.\n", pop_final_top10_recommendation_df)
    else: print('\nHello new user, recommendations based on popularity model are:\n', pop_final_top10_recommendation_df)
    print("--- Total Popularity based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))
else:
    print('\nRecommendation using Content-Based model...')
    content_based_recommender_model = load_reco_model('contentbasedmodel')
    cb_final_top10_recommendation_df = content_based_recommender_model.recommend_items(REC_FOR_USER_ID, recipes_to_ignore_list, 10)
    print(cb_final_top10_recommendation_df)
    print("--- Total content based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

    print('\nRecommendation using collaborative filtering [SVD Matrix Factorization]...')
    cf_recommender_model = load_reco_model('collaborativemodel')
    cf_final_top10_recommendation_df = cf_recommender_model.recommend_items(REC_FOR_USER_ID, recipes_to_ignore_list, 10)
    print(cf_final_top10_recommendation_df)
    print("--- Total SVD based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))

    print('\nRecommendation using Hybrid model...')
    hybrid_recommender_model = load_reco_model('hybridmodel')
    hybrid_final_top10_recommendation_df = hybrid_recommender_model.recommend_items(REC_FOR_USER_ID, recipes_to_ignore_list, 10)
    print(hybrid_final_top10_recommendation_df)
    print("--- Total Hybrid based recommendation engine execution time is %s min ---" % ((time.time() - start_time) / 60))
sys.argv.clear()