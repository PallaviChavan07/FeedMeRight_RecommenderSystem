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
class ContentBasedRecommenderAll:
    MODEL_NAME = 'CBAll'
    CB_SCORE_RATING_FACTOR = 4.0
    def __init__(self, recipe_df=None, interactions_train_indexed_df=None, user_df=None):
        recipe_ids = recipe_df['recipe_id'].tolist()

        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer_in = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, max_df=0.80, stop_words=stopwords.words('english'))
        self.tfidf_matrix_in = vectorizer_in.fit_transform(recipe_df['ingredients'])
        #vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, max_df=0.80, stop_words=stopwords.words('english'))
        #self.tfidf_matrix = vectorizer.fit_transform( recipe_df['cook_method'] + "" +recipe_df['ingredients'] + "" + recipe_df['diet_labels'])
        vectorizer_ck = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
        self.tfidf_matrix_ck = vectorizer_ck.fit_transform(recipe_df['cook_method'])
        vectorizer_dl = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
        self.tfidf_matrix_dl = vectorizer_dl.fit_transform(recipe_df['diet_labels'])

        self.recipe_ids = recipe_ids
        self.recipe_df = recipe_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.user_df = user_df

        self.user_profiles = self.build_users_profiles()

    def get_model_name(self):
        return self.MODEL_NAME

    def get_item_profile(self, item_id):
        idx = self.recipe_ids.index(item_id)
        item_profile_in = self.tfidf_matrix_in[idx:idx + 1]
        item_profile_ck = self.tfidf_matrix_ck[idx:idx + 1]
        item_profile_dl = self.tfidf_matrix_dl[idx:idx + 1]
        return item_profile_in, item_profile_ck, item_profile_dl

    def get_item_profiles(self, ids):
        item_profiles_list_in = []
        item_profiles_list_ck = []
        item_profiles_list_dl = []
        try:
            for x in ids:
                item_profiles_in, item_profiles_ck, item_profiles_dl = self.get_item_profile(x)
                item_profiles_list_in.append(item_profiles_in)
                item_profiles_list_ck.append(item_profiles_ck)
                item_profiles_list_dl.append(item_profiles_dl)
        except:
            #if ids is just a single item (new user issue)
            item_profiles_in, item_profiles_ck, item_profiles_dl = self.get_item_profile(ids)
            item_profiles_list_in.append(item_profiles_in)
            item_profiles_list_ck.append(item_profiles_ck)
            item_profiles_list_dl.append(item_profiles_dl)

        item_profiles_in = scipy.sparse.vstack(item_profiles_list_in)
        item_profiles_ck = scipy.sparse.vstack(item_profiles_list_ck)
        item_profiles_dl = scipy.sparse.vstack(item_profiles_list_dl)
        return item_profiles_in, item_profiles_ck, item_profiles_dl

    def build_users_profile(self, user_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[user_id]
        try:
            user_interactions_items = interactions_person_df['recipe_id']
        except:
            user_interactions_items = None

        # some users might not have any recipe_id so check for the type
        if not user_interactions_items is None:
            user_item_profiles_in, user_item_profiles_ck, user_item_profiles_dl = self.get_item_profiles(user_interactions_items)
            user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1, 1)

            # Weighted average of item profiles by the interactions strength
            user_item_strengths_weighted_avg_in = np.sum(user_item_profiles_in.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
            user_profile_norm_in = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg_in)
            user_item_strengths_weighted_avg_ck = np.sum(user_item_profiles_ck.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
            user_profile_norm_ck = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg_ck)
            user_item_strengths_weighted_avg_dl = np.sum(user_item_profiles_dl.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
            user_profile_norm_dl = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg_dl)
        else:
            user_profile_norm_in = user_profile_norm_ck = user_profile_norm_dl = None

        return user_profile_norm_in, user_profile_norm_ck, user_profile_norm_dl

    def build_users_profiles(self):
        interactions_indexed_df = self.interactions_train_indexed_df[self.interactions_train_indexed_df['recipe_id'].isin(self.recipe_df['recipe_id'])]
        user_profiles = {}
        for user_id in interactions_indexed_df.index.unique():
            user_profiles_in, user_profiles_ck, user_profiles_dl = self.build_users_profile(user_id, interactions_indexed_df)
            user_profiles[str(user_id)+'_in'] = user_profiles_in
            user_profiles[str(user_id) + '_ck'] = user_profiles_ck
            user_profiles[str(user_id) + '_dl'] = user_profiles_dl
        return user_profiles

    def _get_similar_items_to_user_profile(self, user_id):
        # Computes the cosine similarity between the user profile and all item profiles
        try:
            # Gets the top similar items
            cosine_similarities_in = linear_kernel(self.user_profiles[str(user_id)+'_in'], self.tfidf_matrix_in)
            similar_indices_in = cosine_similarities_in.argsort().flatten()
            cosine_similarities_ck = linear_kernel(self.user_profiles[str(user_id) + '_ck'], self.tfidf_matrix_ck)
            similar_indices_ck = cosine_similarities_ck.argsort().flatten()
            cosine_similarities_dl = linear_kernel(self.user_profiles[str(user_id) + '_dl'], self.tfidf_matrix_dl)
            similar_indices_dl = cosine_similarities_dl.argsort().flatten()

            # Sort the similar items by similarity
            #similar_items = sorted([(self.recipe_ids[i], cosine_similarities_in[0, i]) for i in similar_indices_in], key=lambda x: -x[1])
            for i in similar_indices_in:
                cal_denominator = 1
                # if cosine_similarities_in[0, i] == 0.0: cal_denominator -= 1
                # if cosine_similarities_ck[0, i] == 0.0: cal_denominator -= 1
                # if cosine_similarities_dl[0, i] == 0.0: cal_denominator -= 1
                # if cal_denominator <= 0: cal_denominator = 1
                combined_similarities = (self.recipe_ids[i], (cosine_similarities_in[0, i] + cosine_similarities_ck[0, i] + cosine_similarities_dl[0, i])/cal_denominator)

            similar_items = sorted([combined_similarities], key=lambda x: -x[1])
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

        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'calories', 'diet_labels']]
        # convert similarity score to close to equivalent rating
        #recommendations_df['recStrength'] = (recommendations_df['recStrength'] * self.CB_SCORE_RATING_FACTOR) + 1.0

        recommendations_df = self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)
        return recommendations_df