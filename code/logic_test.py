import scipy
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

import os
from sklearn.model_selection import train_test_split
pd.set_option("display.max_rows", None, "display.max_columns", None)
#data
recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
rating_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
user_df = pd.read_csv(os.path.realpath('../data/clean/users.csv'))
print("With empty recipes = ", len(recipe_df))
recipe_df = recipe_df[recipe_df['clean_ingredients'] != np.nan]
print("Without empty recipes = ", len(recipe_df))
user_df = user_df.head(5)
# valid_users_interaction_df is a subset of rating_df
valid_users_interaction_df = pd.merge(rating_df, user_df, on='user_id', how='inner')
# get unique recipes from merged df
unique_valid_recipes = valid_users_interaction_df.recipe_id.unique()
recipe_df = recipe_df[recipe_df['recipe_id'].isin(unique_valid_recipes)]
merged_df = pd.merge(recipe_df, valid_users_interaction_df, on='recipe_id', how='inner')

interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]
interactions_train_df, interactions_test_df = train_test_split(interactions_df, test_size=0.20)
# print('# interactions on Train set: %d' % len(interactions_train_df))
# print('# interactions on Test set: %d' % len(interactions_test_df))

id_list = recipe_df['recipe_id'].tolist()
#print(user_df.loc[user_df['user_id'] == 9259]['recipe_id'].values)
#id_list = rating_df.loc[rating_df['user_id'] == 9259]['recipe_id'].values.tolist()
print("count # of recipe ids:", len(id_list))
#pick 2 random items from the list
import random
listof2items = random.sample(id_list, 2)
print(listof2items)

vectorizer1 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
diet_tfidf = vectorizer1.fit_transform(recipe_df['diet_labels'])
vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
cook_tfidf = vectorizer2.fit_transform(recipe_df['cook_method'])  # min_df=0.01, max_df=0.80, token_pattern=r'\w{1,}',
vectorizer3 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
ingred_tfidf = vectorizer3.fit_transform(recipe_df['clean_ingredients'].values.astype('U'))
# vectorizer_combined = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),  stop_words=stopwords.words('english'))
# combined_tfidf = vectorizer_combined.fit_transform(recipe_df['clean_ingredients']+" "+recipe_df['cook_method']+" "+recipe_df['diet_labels'] )


#print("diet_tfidf", diet_tfidf)
#print("cook_tfidf", cook_tfidf)
print("diet features== \n",vectorizer1.get_feature_names())
print("cook_tfidf features== \n",vectorizer2.get_feature_names())
print("ingred_tfidf features== ",len(vectorizer3.get_feature_names()), "\n",vectorizer3.get_feature_names())

recipe_ids = id_list
idx = recipe_ids.index(listof2items[0])
item_profile1 = diet_tfidf[idx:idx + 1]
idx = recipe_ids.index(listof2items[1])
item_profile2= diet_tfidf[idx:idx + 1]

idx = recipe_ids.index(listof2items[0])
ck_item_profile1 = cook_tfidf[idx:idx + 1]
idx = recipe_ids.index(listof2items[1])
ck_item_profile2= cook_tfidf[idx:idx + 1]

idx = recipe_ids.index(listof2items[0])
in_item_profile1 = ingred_tfidf[idx:idx + 1]
idx = recipe_ids.index(listof2items[1])
in_item_profile2= ingred_tfidf[idx:idx + 1]

# idx = recipe_ids.index(listof2items[0])
# combined_item_profile1 = combined_tfidf[idx:idx + 1]
# idx = recipe_ids.index(listof2items[1])
# combined_item_profile2= combined_tfidf[idx:idx + 1]


dl_user_item_profiles = []
ck_user_item_profiles = []
in_user_item_profiles = []
combined_user_item_profiles = []

dl_user_item_profiles.append(item_profile1)
dl_user_item_profiles.append(item_profile2)
ck_user_item_profiles.append(ck_item_profile1)
ck_user_item_profiles.append(ck_item_profile2)
in_user_item_profiles.append(in_item_profile1)
in_user_item_profiles.append(in_item_profile2)
# combined_user_item_profiles.append(combined_item_profile1)
# combined_user_item_profiles.append(combined_item_profile2)

user_item_strengths = [4,5]
user_item_strengths = np.array(user_item_strengths).reshape(-1, 1)

item_profiles = scipy.sparse.vstack(dl_user_item_profiles)
ck_item_profiles = scipy.sparse.vstack(ck_user_item_profiles)
in_item_profiles = scipy.sparse.vstack(in_user_item_profiles)
#combined_item_profiles = scipy.sparse.vstack(combined_user_item_profiles)
# Weighted average of item profiles by the interactions strength
user_item_strengths_weighted_avg = np.sum(item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
#print("user_profile_norm", user_profile_norm)

ck_user_item_strengths_weighted_avg = np.sum(ck_item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
ck_user_profile_norm = sklearn.preprocessing.normalize(ck_user_item_strengths_weighted_avg)
#print("ck_user_profile_norm", ck_user_profile_norm)

in_user_item_strengths_weighted_avg = np.sum(in_item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
in_user_profile_norm = sklearn.preprocessing.normalize(in_user_item_strengths_weighted_avg)
#print("in_user_profile_norm", in_user_profile_norm)
#combined_user_item_strengths_weighted_avg = np.sum(combined_item_profiles.multiply(user_item_strengths),
#                                          axis=0) / np.sum(user_item_strengths)
#combined_user_profile_norm = sklearn.preprocessing.normalize(combined_user_item_strengths_weighted_avg)
#print("combined_user_profile_norm", combined_user_profile_norm)


diet_cosine_sim = linear_kernel(user_profile_norm, diet_tfidf)
cook_cosine_sim = linear_kernel(ck_user_profile_norm, cook_tfidf)
in_cosine_sim = linear_kernel(in_user_profile_norm, ingred_tfidf)
#combined_cosin_sim = linear_kernel(combined_user_profile_norm, combined_tfidf)
#print("diet_cosine_sim = \n", (diet_cosine_sim))
#print("cook_cosine_sim = \n", cook_cosine_sim)
#print("in_cosine_sim = \n", in_cosine_sim)
#print("combined_cosin_sim = \n", combined_cosin_sim)

diet_similar_indices = diet_cosine_sim.argsort().flatten()
diet_similar_items = sorted([(id_list[i], diet_cosine_sim[0, i]) for i in diet_similar_indices], key=lambda x: -x[1])
#print("diet_similar_items = ", diet_similar_items)

cook_similar_indices = cook_cosine_sim.argsort().flatten()
cook_similar_items = sorted([(id_list[i], cook_cosine_sim[0, i]) for i in cook_similar_indices], key=lambda x: -x[1])
#print("cook_similar_items = ", cook_similar_items)

in_similar_indices = in_cosine_sim.argsort().flatten()
in_similar_items = sorted([(id_list[i], in_cosine_sim[0, i]) for i in in_similar_indices], key=lambda x: -x[1])
#print("in_similar_items = ", in_similar_items)

# combined_similar_indices = combined_cosin_sim.argsort().flatten()
# combined_similar_items = sorted([(id_list[i], combined_cosin_sim[0, i]) for i in combined_similar_indices], key=lambda x: -x[1])



average_similar_items = sorted([(id_list[i], (cook_cosine_sim[0, i]+ diet_cosine_sim[0,i]) /2 ) for i in cook_similar_indices], key=lambda x: -x[1])
#print("average_similar_items = ", average_similar_items)

lastaverage_similar_items = sorted([(id_list[i], (cook_cosine_sim[0, i]+ diet_cosine_sim[0,i] + in_cosine_sim[0,i]) /3 ) for i in cook_similar_indices], key=lambda x: -x[1])
# print("lastaverage_similar_items = ", lastaverage_similar_items)
# print("combined_similar_items = ", combined_similar_items)

