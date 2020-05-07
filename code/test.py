from statistics import mean

import scipy
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

recipe_df =pd.DataFrame(columns=['recipe_id', 'diet_labels'])
id_list = [1,2,3,4,5]
diet_list = ["lowcarb highprotien","lowcarb","lowsodium","lowfat","lowfat lowcarb"]
ingred_list = ["Parmesan cheese^ground black pepper^garlic powder^frozen puff pastry thawed^egg white",
             "unsalted butter^chopped onion^cornmeal^all-purpose flour^white sugar^baking powder^salt^baking soda^buttermilk^eggs^shredded pepperjack cheese^frozen corn kernels^roasted marinated red bell peppers^chopped fresh basil",
             "hot water^margarine^white sugar^salt^cold water^active dry yeast^all-purpose flour^eggs",
             "white sugar^vegetable oil^eggs^sifted all-purpose flour^baking soda^salt^ground cinnamon^ground nutmeg^water^cooked and mashed sweet potatoes^chopped pecans",
             "active dry yeast^lukewarm milk^white sugar^unbleached all-purpose flour^salt^butter"]

# ingred_list = ["parmesan cheese pepper garlic puff pastry egg",
#              "butter onion cornmeal all-purpose flour sugar baking baking buttermilk eggs pepperjack cheese corn kernels marinated bell peppers basil",
#              "margarine sugar yeast all-purpose flour eggs",
#              "sugar vegetable oil eggs sifted all-purpose flour baking cinnamon nutmeg and sweet potatoes pecans",
#              "yeast lukewarm milk sugar all-purpose flour butter"]
cook_methods = ["bake","fry","fry blanch","steam boil","steam bake"]
recipe_df['recipe_id'] = id_list
recipe_df['diet_labels'] = diet_list
recipe_df['cook_methods'] = cook_methods
recipe_df['ingredients'] = ingred_list

vectorizer1 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
diet_tfidf = vectorizer1.fit_transform(recipe_df['diet_labels'])
vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stopwords.words('english'))
cook_tfidf = vectorizer2.fit_transform(recipe_df['cook_methods'])  # min_df=0.01, max_df=0.80,
vectorizer3 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),  stop_words=stopwords.words('english'))
ingred_tfidf = vectorizer3.fit_transform(recipe_df['ingredients'])
vectorizer_combined = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),  stop_words=stopwords.words('english'))
combined_tfidf = vectorizer_combined.fit_transform(recipe_df['ingredients']+" "+recipe_df['cook_methods']+" "+recipe_df['diet_labels'] )


print("tfidf_matrix", diet_tfidf)
print("cook_tfidf", cook_tfidf)
print("diet features== \n",vectorizer1.get_feature_names())
print("cook_tfidf features== \n",vectorizer2.get_feature_names())
print("ingred_tfidf features== \n",vectorizer3.get_feature_names())

recipe_ids = id_list
idx = recipe_ids.index(1)
item_profile1 = diet_tfidf[idx:idx + 1]
idx = recipe_ids.index(2)
item_profile2= diet_tfidf[idx:idx + 1]

idx = recipe_ids.index(1)
ck_item_profile1 = cook_tfidf[idx:idx + 1]
idx = recipe_ids.index(2)
ck_item_profile2= cook_tfidf[idx:idx + 1]

idx = recipe_ids.index(1)
in_item_profile1 = ingred_tfidf[idx:idx + 1]
idx = recipe_ids.index(2)
in_item_profile2= ingred_tfidf[idx:idx + 1]

idx = recipe_ids.index(1)
combined_item_profile1 = combined_tfidf[idx:idx + 1]
idx = recipe_ids.index(2)
combined_item_profile2= combined_tfidf[idx:idx + 1]


user_item_profiles = []
ck_user_item_profiles = []
in_user_item_profiles = []
combined_user_item_profiles = []

user_item_profiles.append(item_profile1)
user_item_profiles.append(item_profile2)
ck_user_item_profiles.append(ck_item_profile1)
ck_user_item_profiles.append(ck_item_profile2)
in_user_item_profiles.append(in_item_profile1)
in_user_item_profiles.append(in_item_profile2)
combined_user_item_profiles.append(combined_item_profile1)
combined_user_item_profiles.append(combined_item_profile2)

user_item_strengths = [4,5]
user_item_strengths = np.array(user_item_strengths).reshape(-1, 1)

item_profiles = scipy.sparse.vstack(user_item_profiles)
ck_item_profiles = scipy.sparse.vstack(ck_user_item_profiles)
in_item_profiles = scipy.sparse.vstack(in_user_item_profiles)
combined_item_profiles = scipy.sparse.vstack(combined_user_item_profiles)
# Weighted average of item profiles by the interactions strength
user_item_strengths_weighted_avg = np.sum(item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
print("user_profile_norm", user_profile_norm)

ck_user_item_strengths_weighted_avg = np.sum(ck_item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
ck_user_profile_norm = sklearn.preprocessing.normalize(ck_user_item_strengths_weighted_avg)
print("ck_user_profile_norm", ck_user_profile_norm)

in_user_item_strengths_weighted_avg = np.sum(in_item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
in_user_profile_norm = sklearn.preprocessing.normalize(in_user_item_strengths_weighted_avg)
print("in_user_profile_norm", in_user_profile_norm)
combined_user_item_strengths_weighted_avg = np.sum(combined_item_profiles.multiply(user_item_strengths),
                                          axis=0) / np.sum(user_item_strengths)
combined_user_profile_norm = sklearn.preprocessing.normalize(combined_user_item_strengths_weighted_avg)
print("combined_user_profile_norm", combined_user_profile_norm)


diet_cosine_sim = linear_kernel(user_profile_norm, diet_tfidf)
cook_cosine_sim = linear_kernel(ck_user_profile_norm, cook_tfidf)
in_cosine_sim = linear_kernel(in_user_profile_norm, ingred_tfidf)
combined_cosin_sim = linear_kernel(combined_user_profile_norm, combined_tfidf)
print("diet_cosine_sim = \n", diet_cosine_sim)
print("cook_cosine_sim = \n", cook_cosine_sim)
print("in_cosine_sim = \n", in_cosine_sim)
print("combined_cosin_sim = \n", combined_cosin_sim)

diet_similar_indices = diet_cosine_sim.argsort().flatten()
diet_similar_items = sorted([(id_list[i], diet_cosine_sim[0, i]) for i in diet_similar_indices], key=lambda x: -x[1])
print("diet_similar_items = ", diet_similar_items)

cook_similar_indices = cook_cosine_sim.argsort().flatten()
cook_similar_items = sorted([(id_list[i], cook_cosine_sim[0, i]) for i in cook_similar_indices], key=lambda x: -x[1])
print("cook_similar_items = ", cook_similar_items)

in_similar_indices = in_cosine_sim.argsort().flatten()
in_similar_items = sorted([(id_list[i], in_cosine_sim[0, i]) for i in in_similar_indices], key=lambda x: -x[1])
print("in_similar_items = ", in_similar_items)

combined_similar_indices = combined_cosin_sim.argsort().flatten()
combined_similar_items = sorted([(id_list[i], combined_cosin_sim[0, i]) for i in combined_similar_indices], key=lambda x: -x[1])



average_similar_items = sorted([(id_list[i], (cook_cosine_sim[0, i]+ diet_cosine_sim[0,i]) /2 ) for i in cook_similar_indices], key=lambda x: -x[1])
print("average_similar_items = ", average_similar_items)

lastaverage_similar_items = sorted([(id_list[i], (cook_cosine_sim[0, i]+ diet_cosine_sim[0,i] + in_cosine_sim[0,i]) /3 ) for i in cook_similar_indices], key=lambda x: -x[1])
print("lastaverage_similar_items = ", lastaverage_similar_items)
print("combined_similar_items = ", combined_similar_items)