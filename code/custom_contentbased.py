import numpy as np
import scipy
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

########################################## CONTENT BASED ##########################################
class ContentBasedRecommender:
    MODEL_NAME = 'ContentBased'
    def __init__(self, recipe_df=None, interactions_full_indexed_df=None):
        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.003, max_df=0.80, stop_words=stopwords.words('english'))
        recipe_ids = recipe_df['recipe_id'].tolist()

        self.tfidf_matrix = vectorizer.fit_transform(recipe_df['recipe_name'] + "" + recipe_df['ingredients'])
        self.tfidf_feature_names = vectorizer.get_feature_names()
        self.recipe_ids = recipe_ids
        self.recipe_df = recipe_df
        self.interactions_full_indexed_df = interactions_full_indexed_df

        self.user_profiles = self.build_users_profiles()
        print("\nTotal User Profiles: ", len(self.user_profiles))
        # print(user_profiles)
        # myprofile = user_profiles[3324846]
        # print(myprofile.shape)
        # print(pd.DataFrame(sorted(zip(tfidf_feature_names, user_profiles[3324846].flatten().tolist()), key=lambda x: -x[1])[:20], columns=['token', 'relevance']))
        # myprofile = user_profiles[682828]
        # print(myprofile.shape)

    def get_model_name(self):
        return self.MODEL_NAME

    def get_recipes_interacted(self, user_id):
        # Get the user's data and merge in the information.
        try:
            interacted_items = self.interactions_full_indexed_df.loc[user_id]
        except:
            interacted_items = None
        return interacted_items

    def cb_evaluate_model_for_user(self, user_id, users_cb_recs_df, k=5):
        if users_cb_recs_df is None:
            return {'p_recall': 0, 'a_recall': 0, 'user_top_k_recos_count': 0, 'user_interated_relevant_count': 0,
                    'k': k}

        # get top k recos for the user from the complete users_cb_recs_df
        user_top_k_recos = users_cb_recs_df.head(k)

        # get recipes already interacted by user
        user_interact_recipes_df = self.get_recipes_interacted(user_id)
        # print("user_interact_recipes_df: ", len(user_interact_recipes_df), " for user_id ", user_id)

        # filter out recipes with rating > 3.5 which is our threshold for good vs bad recipes
        user_interated_relevant_df = user_interact_recipes_df.loc[user_interact_recipes_df['rating'] >= 3.5]
        # print("user_interated_relevant_df: ", len(user_interated_relevant_df))

        # merge top k recommended recipes with filtered user interacted recipes to get relevant recommended
        relevant_and_reco_items_df = user_top_k_recos.merge(user_interated_relevant_df, how='inner', on='recipe_id')
        # print("relevant_and_reco_items_df:\n", relevant_and_reco_items_df)

        user_top_k_recos_count = len(user_top_k_recos)
        p_recall = len(relevant_and_reco_items_df) / user_top_k_recos_count if user_top_k_recos_count != 0 else 1
        # print("Pallavi dumb recall", p_recall)

        # Recall@K: Proportion of relevant items that are recommended
        n_rel_and_rec_k = len(relevant_and_reco_items_df)
        n_rel = len(user_interated_relevant_df)
        a_recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        # print("amod yet to correct but dumb recall", a_recall)

        person_metrics = {'p_recall': p_recall,
                          'a_recall': a_recall,
                          'user_top_k_recos_count': user_top_k_recos_count,
                          'user_interated_relevant_count': n_rel,
                          'k': k}

        # print(person_metrics)
        return person_metrics

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
            cosine_similarities = cosine_similarity(self.user_profiles[user_id], self.tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()
            #print("Take only top ", len(similar_indices), "similar items")
            # Sort the similar items by similarity
            similar_items = sorted([(self.recipe_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        except:
            return None

        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        try:
            similar_items = self._get_similar_items_to_user_profile(user_id)
        except:
            return None
        # early exit
        if similar_items is None: return None

        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['recipe_id', 'recStrength']).head(topn)
        #recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id']]
        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions']]

        #print(recommendations_df.shape)
        return recommendations_df