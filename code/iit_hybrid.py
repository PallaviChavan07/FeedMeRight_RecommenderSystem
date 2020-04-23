import pandas as pd
import numpy as np
import scipy
import sklearn
import surprise
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset, SVD

class hybrid(object):
    def __init__(self, user_id, merged_df, ratings):
        self.user_id = user_id
        self.md = merged_df
        self.ratings = ratings
        #print(ratings[(ratings['user_id'] == user_id)][['user_id', 'recipe_id', 'rating']])

        #self.popularity_rating = self.popularity(self.md)
        self.collaborative_rating = self.collaborative(self.ratings, self.user_id)
        self.content_rating = self.content_based(self.md, self.ratings, self.user_id)
        self.final_hybrid(self.md, self.collaborative_rating, self.content_rating, self.user_id)

    ### Collaborative ##
    def collaborative(self, ratings, user_id):
        reader = Reader()
        temp_ratings = ratings

        data = Dataset.load_from_df(temp_ratings[['user_id', 'recipe_id', 'rating']], reader)
        trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)
        algo = SVD(random_state=0)
        algo.fit(trainSet)
        predictions = algo.test(testSet)
        rmseSVD = surprise.accuracy.rmse(predictions, verbose=False)

        for uid, iid, true_r, est, _ in predictions:
            if uid == user_id:
                temp_ratings.loc[len(temp_ratings)] = [uid, iid, est]

        cb = temp_ratings[(temp_ratings['user_id'] == user_id)][['recipe_id', 'rating']]
        cb.columns = ['recipe_id', 'cf_rating']

        print("Col in CB rec df: ", cb.columns.tolist())
        print("Total CF SVD recos: ", len(cb))
        print("rmseSVD: ", rmseSVD)

        return cb

    ##### CONTENT ######
    def content_based(self, recipe_df, ratings, user_id):
        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.003, max_df=0.5, max_features=5000, stop_words='english')
        item_ids = recipe_df['recipe_id'].tolist()
        tfidf_matrix = vectorizer.fit_transform(recipe_df['ingredients'])

        def get_item_profile(item_id):
            idx = item_ids.index(item_id)
            item_profile = tfidf_matrix[idx:idx + 1]
            return item_profile

        def get_item_profiles(ids):
            item_profiles_list = [get_item_profile(x) for x in ids]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles

        def build_users_profile(user_id, interactions_indexed_df):
            interactions_person_df = interactions_indexed_df.loc[user_id]
            try:
                user_interactions_items = interactions_person_df['recipe_id']
            except:
                user_interactions_items = None

            # some users might not have any recipe_id so check for the type
            if type(user_interactions_items) == pd.Series:
                user_item_profiles = get_item_profiles(interactions_person_df['recipe_id'])
                user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1, 1)
                # Weighted average of item profiles by the interactions strength
                user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths),
                                                          axis=0) / np.sum(user_item_strengths)
                user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
            else:
                user_profile_norm = None

            return user_profile_norm

        def build_users_profiles():
            interactions_indexed_df = ratings[ratings['recipe_id'].isin(recipe_df['recipe_id'])].set_index('user_id')
            user_profiles = {}
            for user_id in interactions_indexed_df.index.unique():
                user_profiles[user_id] = build_users_profile(user_id, interactions_indexed_df)
            return user_profiles

        user_profiles = build_users_profiles()
        print("\nTotal User Profiles: ", len(user_profiles))
        myprofile = user_profiles[3324846]

        try:
            cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()[-100:]
            # Sort the similar items by similarity
            similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        except:
            return None

        # Ignores items the user has already interacted
        similar_items_filtered = list(similar_items)
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['recipe_id', 'cb_rating']).head(10)

        print("Col in CB rec df: ", recommendations_df.columns.tolist())
        print("Total CB recos: ", len(recommendations_df))
        #print("rmseSVD: ", rmseSVD)

        return recommendations_df

    def final_hybrid(self, md, collaborative_rating, content_rating, user_id):
        hyb = md[['recipe_id', 'recipe_name', 'ingredients', 'rating']]
        hyb = hyb.merge(collaborative_rating, on='recipe_id')
        hyb = hyb.merge(content_rating, on='recipe_id')

        cb_ensemble_weight = 0.4
        cf_ensemble_weight = 0.6

        def weighted_rating(x):
            cf = x['cf_rating'] * cf_ensemble_weight
            cb = x['cb_rating'] * cb_ensemble_weight
            return cf + cb

        hyb['hyb_rating'] = hyb.apply(weighted_rating, axis=1)
        hyb = hyb.sort_values('hyb_rating', ascending=False).head(999)
        #hyb.columns = ['Recipe ID', 'Recipe Name', 'Ingredients', 'Rating', 'SVD Rating', 'ContentBased Rating', 'Hybrid Rating']

        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print("Recommendations for user with id: ", user_id)
        hyb = hyb[['recipe_id', 'recipe_name', 'rating', 'cf_rating', 'cb_rating', 'hyb_rating']]
        hyb = hyb.set_index('recipe_id')
        #print(hyb.index.unique().values)
        print(hyb)


recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')
train_rating_df = train_rating_df[['user_id', 'recipe_id', 'rating']]
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
h = hybrid(3324846, merged_df, train_rating_df)
