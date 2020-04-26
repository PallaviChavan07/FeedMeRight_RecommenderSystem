import numpy as np
import scipy
import pandas as pd
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#New Code Start
test_user_id = 9259
#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 50
pd.set_option("display.max_rows", None, "display.max_columns", None)

recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
#print(recipe_df.head(5))
train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')
merged_df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
#interactions_df = interactions_df.set_index('user_id')
#print(interactions_df.head(5))
#print("\nUser Id [", test_user_id, "] details: \n", interactions_df.loc[interactions_df['user_id'] == test_user_id], "\n")

users_interactions_count_df = interactions_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
#users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 0].reset_index()[['user_id']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, how = 'right', left_on = 'user_id', right_on = 'user_id')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'recipe_id'])['rating'].sum().reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, test_size=0.20)
print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
#New Code End

#Indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')
#print(interactions_test_indexed_df.loc[test_user_id])

def get_items_interacted(user_id, interactions_df):
    # Get the user's data and merge in the information.
    try:
        interacted_items = interactions_df.loc[user_id]['recipe_id']
    except:
        interacted_items = None

    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

class ModelEvaluator:
    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted(user_id, interactions_full_indexed_df)
        all_items = set(recipe_df['recipe_id'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, user_id):
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_testset['recipe_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['recipe_id'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['recipe_id'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(user_id, items_to_ignore=get_items_interacted(user_id, interactions_train_indexed_df),topn=10000000000)
        #if user_id == test_user_id: print("person_recs_df \n", person_recs_df[['user_id', 'recipe_id', 'cb_rating']])

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            if not person_recs_df is None:
                valid_recs_df = person_recs_df[person_recs_df['recipe_id'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['recipe_id'].values
            else:
                #this way we can still get person_metrics
                valid_recs = None

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_cnt': hits_at_5_count,
                          'hits@10_cnt': hits_at_10_count,
                          'interacted_cnt': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}

        #if person_recs_df is None: print(person_metrics)
        return person_metrics, person_recs_df

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        users_recs_df = pd.DataFrame()
        for idx, user_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics, singleuser_recs_df = self.evaluate_model_for_user(model, user_id)
            person_metrics['_user_id'] = user_id
            people_metrics.append(person_metrics)
            if not singleuser_recs_df is None: users_recs_df = pd.concat([users_recs_df, singleuser_recs_df[['user_id', 'recipe_id', 'cb_rating']]])
        print('%d users processed' % idx)
        print('users_recs_list shape: ', users_recs_df.shape)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_cnt', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_cnt'].sum() / float(detailed_results_df['interacted_cnt'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_cnt'].sum() / float(detailed_results_df['interacted_cnt'].sum())

        global_metrics = {'modelName': model.get_model_name(), 'recall@5': global_recall_at_5, 'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df, users_recs_df

model_evaluator = ModelEvaluator()

########################################## CONTENT BASED ##########################################
#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.003, max_df=0.5, stop_words=stopwords_list)

item_ids = recipe_df['recipe_id'].tolist()
tfidf_matrix = vectorizer.fit_transform(recipe_df['recipe_name'] + "" + recipe_df['ingredients'])
tfidf_feature_names = vectorizer.get_feature_names()

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

    #some users might not have any recipe_id so check for the type
    if type(user_interactions_items) == pd.Series:
        user_item_profiles = get_item_profiles(interactions_person_df['recipe_id'])
        user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    else:
        user_profile_norm = None

    return user_profile_norm

def build_users_profiles():
    interactions_indexed_df = interactions_train_df[interactions_train_df['recipe_id'].isin(recipe_df['recipe_id'])].set_index('user_id')
    user_profiles = {}
    for user_id in interactions_indexed_df.index.unique():
        user_profiles[user_id] = build_users_profile(user_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()
print("\nTotal User Profiles: ", len(user_profiles))
#print(user_profiles)
myprofile = user_profiles[3324846]
#print(myprofile.shape)
#print(pd.DataFrame(sorted(zip(tfidf_feature_names, user_profiles[3324846].flatten().tolist()), key=lambda x: -x[1])[:20], columns=['token', 'relevance']))
#myprofile = user_profiles[682828]
#print(myprofile.shape)

class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, user_id):
        # Computes the cosine similarity between the user profile and all item profiles
        try:
            cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)
            # Gets the top similar items
            similar_indices = cosine_similarities.argsort().flatten()
            #print("Take only top ", len(similar_indices), "similar items")
            # Sort the similar items by similarity
            similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
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
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['recipe_id', 'cb_rating']).head(topn)
        if self.items_df is None:
            raise Exception('"items_df" is required in verbose mode')

        recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='recipe_id', right_on='recipe_id')[
            ['cb_rating', 'recipe_id', 'recipe_name', 'ingredients', 'nutritions']]
        recommendations_df['user_id'] = user_id

        return recommendations_df

content_based_recommender_model = ContentBasedRecommender(recipe_df)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print('\nEvaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df, users_cb_recs_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('Global metrics:\n%s' % cb_global_metrics)
print("\n", cb_detailed_results_df.head(5))
#print(users_cb_recs_df.head(5))
#if users_cb_recs_df is not None: print("Printing recommendation df for ", test_user_id, " :\n", users_cb_recs_df.loc[users_cb_recs_df['user_id'] == test_user_id].head(10))

########################################## SURPRISE SVD ##########################################
from surprise import SVDpp
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from code import Evaluators, Recipe_Reco_SingleUser, Top5_Recipe_Reco_PerUser

def SVDplusplus():
    # Read Data
    recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
    train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')

    # stats after duplicates (if any) removal
    train_rating_df.drop_duplicates(inplace=True)
    #print('we have', train_rating_df.shape[0], 'ratings')
    #print('the number of unique users we have is:', len(train_rating_df.user_id.unique()))
    #print('the number of unique recipes we have is:', len(train_rating_df.recipe_id.unique()))
    #print("The median user rated %d books." % train_rating_df.user_id.value_counts().median())
    #print('The max rating is: %d' % train_rating_df.rating.max(), "and the min rating is: %d" % train_rating_df.rating.min())
    df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)

    algo = SVDpp()
    algo.fit(trainSet)
    predictions = algo.test(testSet)
    #print("SURPRISE SVD PREDS\n", predictions)

    precisions, recalls = Evaluators.precision_recall_at_k(predictions)
    precisionAt10 = sum(prec for prec in precisions.values()) / len(precisions)
    recallAt10 = sum(rec for rec in recalls.values()) / len(recalls)
    precisions, recalls = Evaluators.precision_recall_at_k(predictions, k=5)
    precisionAt5 = sum(prec for prec in precisions.values()) / len(precisions)
    recallAt5 = sum(rec for rec in recalls.values()) / len(recalls)

    svd_metrics = {'precision@5': precisionAt5, 'precision@10': precisionAt10,
                   'recall@5': recallAt5, 'recall@10': recallAt10}

    svd_df = pd.DataFrame(columns = ['user_id', 'recipe_id', 'svd_rating'])
    for uid, iid, true_r, est, _ in predictions:
        tempdf = pd.DataFrame([[uid, iid, est]], columns=['user_id', 'recipe_id', 'svd_rating'])
        svd_df = pd.concat([svd_df, tempdf])

    print('users_recs_list shape: ', svd_df.shape)

    #Display Results
    #Top5_Recipe_Reco_PerUser.DisplayResults(predictions)
    #Recipe_Reco_SingleUser.GetSingleUserRecipeReco(df, algo, test_user_id)
    return svd_metrics, svd_df

print('\nEvaluating SVD model...')
svd_metrics, users_svd_pred_df = SVDplusplus()
print('SVD metrics:\n', svd_metrics)
#print(users_svd_pred_df.head(5))
#if users_svd_pred_df is not None: print("Printing pred df for ", test_user_id, " :\n", users_svd_pred_df.loc[users_svd_pred_df['user_id'] == test_user_id].head(10))

########################################## HYBRID WEIGHTED RATING ##########################################
print('\nRunning hybrid model...')
#because content base DF has almsot all recipe ids (for any given user) while SVD only computes few, "OUTER JOIN" cb df on svd df.
singleuser_svd_df = users_svd_pred_df.loc[users_svd_pred_df['user_id'] == test_user_id]
singleuser_cb_df = users_cb_recs_df.loc[users_cb_recs_df['user_id'] == test_user_id]
hyb = singleuser_cb_df.merge(singleuser_svd_df, how='outer', left_on='recipe_id', right_on='recipe_id').fillna(0.0)
#hyb = users_cb_recs_df.merge(users_svd_pred_df, how='outer', left_on='recipe_id', right_on='recipe_id').fillna(0.0)
#print(hyb.columns.tolist())

def weighted_rating(x):
    cf = x['svd_rating'] * 0.7
    cb = x['cb_rating'] * 0.3
    return cf + cb

hyb['hyb_rating'] = hyb.apply(weighted_rating, axis=1)
hyb = hyb.sort_values('hyb_rating', ascending=False)

# after all the work is done, drop already rated recipes ids by user_id
user_alreadyRatedRecipeId = train_rating_df[(train_rating_df['user_id'] == test_user_id)][['recipe_id']]
# Get all indexes
indexNames = hyb[hyb['recipe_id'].isin(user_alreadyRatedRecipeId['recipe_id'].tolist())].index
print("user_alreadyRatedBookId: ", user_alreadyRatedRecipeId['recipe_id'].tolist())
# Delete these row indexes from dataFrame
hyb.drop(indexNames, inplace=True)
#hyb.drop_duplicates('recipe_id', inplace=True)

print("\n# of hybrid ratings computed:", len(hyb['hyb_rating']))
print("Show ratings grid: \n", hyb.head(25))
