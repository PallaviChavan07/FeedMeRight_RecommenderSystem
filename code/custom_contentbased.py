import numpy as np
import scipy
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler

########################################## CONTENT BASED ##########################################
class ContentBasedRecommender:
    MODEL_NAME = 'ContentBased'
    CB_SCORE_RATING_FACTOR = 4.0
    def __init__(self, recipe_df=None, interactions_full_indexed_df=None, user_df=None):
        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, max_df=0.80, stop_words=stopwords.words('english'))
        recipe_ids = recipe_df['recipe_id'].tolist()

        self.tfidf_matrix = vectorizer.fit_transform( recipe_df['cook_method'] + "" +recipe_df['ingredients'] + "" + recipe_df['diet_labels'])
        #self.tfidf_matrix = vectorizer.fit_transform(recipe_df['clean_ingredients'])
        #self.tfidf_matrix = vectorizer.fit_transform(recipe_df['ingredients'])

        self.tfidf_feature_names = vectorizer.get_feature_names()
        self.recipe_ids = recipe_ids
        self.recipe_df = recipe_df
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.user_df = user_df

        self.user_profiles = self.build_users_profiles()
        #print("\nTotal User Profiles: ", len(self.user_profiles))
        # print(user_profiles)
        # myprofile = user_profiles[3324846]
        # print(myprofile.shape)
        # print(pd.DataFrame(sorted(zip(tfidf_feature_names, user_profiles[3324846].flatten().tolist()), key=lambda x: -x[1])[:20], columns=['token', 'relevance']))
        # myprofile = user_profiles[682828]
        # print(myprofile.shape)

    def get_model_name(self):
        return self.MODEL_NAME

    def get_item_profile(self, item_id):
        idx = self.recipe_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx + 1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(self, user_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[user_id]
        try:
            user_interactions_items = interactions_person_df['recipe_id']
        except:
            user_interactions_items = None

        # some users might not have any recipe_id so check for the type
        if type(user_interactions_items) == pd.Series:
            user_item_profiles = self.get_item_profiles(interactions_person_df['recipe_id'])
            user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1, 1)
            # Weighted average of item profiles by the interactions strength
            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths),
                                                      axis=0) / np.sum(user_item_strengths)
            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
        else:
            user_profile_norm = None

        return user_profile_norm

    def build_users_profiles(self):
        interactions_indexed_df = self.interactions_full_indexed_df[
            self.interactions_full_indexed_df['recipe_id'].isin(self.recipe_df['recipe_id'])]
        user_profiles = {}
        for user_id in interactions_indexed_df.index.unique():
            user_profiles[user_id] = self.build_users_profile(user_id, interactions_indexed_df)
        return user_profiles

    def _get_similar_items_to_user_profile(self, user_id):
        # Computes the cosine similarity between the user profile and all item profiles
        try:
            #cosine_similarities = cosine_similarity(self.user_profiles[user_id], self.tfidf_matrix)
            cosine_similarities = linear_kernel(self.user_profiles[user_id], self.tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()
            #print("Take only top ", len(similar_indices), "similar items")
            # Sort the similar items by similarity
            similar_items = sorted([(self.recipe_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        except:
            return None

        return similar_items

    def get_recommendation_for_user_calorie_count(self, cal_rec_df, user_id):
        # print("CB: Before calories filter = ", recommendations_df.shape)
        # get calories required for user
        user_calories_per_day = self.user_df.loc[self.user_df['user_id'] == user_id]['calories_per_day'].values
        # print("CB: user calories per day", user_calories_per_day, type(user_calories_per_day), user_calories_per_day[0])
        # divide calories into 1/3rd part
        user_calories = user_calories_per_day[0] / 3
        # consider only those recipes which have calories less than required calories for that user
        cal_rec_df = cal_rec_df[cal_rec_df['calories'] <= user_calories]
        # print("CB: After calories filter = ", recommendations_df.shape)
        return cal_rec_df

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        try:
            similar_items = self._get_similar_items_to_user_profile(user_id)
        except:
            return None
        # early exit
        if similar_items is None: return None

        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['recipe_id', 'recStrength']).head(topn)

        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'calories', 'diet_labels']]
        # convert similarity score to close to equivalent rating
        recommendations_df['recStrength'] = (recommendations_df['recStrength'] * self.CB_SCORE_RATING_FACTOR) + 1.0

        recommendations_df = self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)
        return recommendations_df