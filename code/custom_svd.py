import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

########################################## COLLABORATIVE FILTERING BASED ##########################################
class CFRecommender:
    MODEL_NAME = 'Collaborative SVD Matrix'
    # The number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 50
    def __init__(self, recipe_df=None, interactions_train_df=None, interactions_full_indexed_df=None, interactions_train_indexed_df=None, interactions_test_indexed_df=None, user_df=None):
        # Creating a sparse pivot table with users in rows and items in columns
        interactions_train_indexed_df = interactions_train_indexed_df.reset_index()
        users_items_pivot_matrix_df = interactions_train_indexed_df.pivot(index='user_id', columns='recipe_id', values='rating').fillna(0)
        users_items_pivot_matrix_df.head(10)

        users_items_pivot_matrix = users_items_pivot_matrix_df
        #print(users_items_pivot_matrix[:10])
        users_ids = list(users_items_pivot_matrix_df.index)
        #print(users_ids[:10])
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        # Performs matrix factorization of the original user item matrix
        if min(users_items_pivot_sparse_matrix.shape) < self.NUMBER_OF_FACTORS_MF: self.NUMBER_OF_FACTORS_MF = 1
        #print("NUMBER_OF_FACTORS_MF=", self.NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=self.NUMBER_OF_FACTORS_MF)
        # print(U.shape)
        # print(Vt.shape)
        sigma = np.diag(sigma)
        # print(sigma.shape)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        # print(all_user_predicted_ratings)

        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                    all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        # Converting the reconstructed matrix back to a Pandas dataframe
        cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns, index=users_ids).transpose()
        # print("CF log:", cf_preds_df.head(5))
        # print("CF log:", len(cf_preds_df.columns))

        self.cf_predictions_df = cf_preds_df
        self.recipe_df = recipe_df
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.user_df = user_df

    def get_model_name(self):
        return self.MODEL_NAME

    def get_recommendation_for_user_calorie_count(self, cal_rec_df, user_id):
        # print("CF: Before calories filter = ", recommendations_df.shape)
        # get calories required for user
        user_calories_per_day = self.user_df.loc[self.user_df['user_id'] == user_id]['calories_per_day'].values
        # print("CF: user calories per day", user_calories_per_day, type(user_calories_per_day), user_calories_per_day[0])
        # divide calories into 1/3rd part
        user_calories = user_calories_per_day[0] / 3
        # consider only those recipes which have calories less than required calories for that user
        cal_rec_df = cal_rec_df[cal_rec_df['calories'] <= user_calories]
        # print("CF: After calories filter = ", recommendations_df.shape)
        return cal_rec_df

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Get and sort the user's predictions
        try:
            sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        except:
            return None

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['recipe_id'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)
        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'calories', 'diet_labels']]
        recommendations_df = self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)
        return recommendations_df
