import nltk
import pandas as pd
import json
import ast
import numpy as np
import os
from random import seed
from random import randint
nltk.download(['stopwords','wordnet'])
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
stop_words = stopwords.words('english')
import spacy

sedentary = 1
lightly_active = 2
moderately_active = 3
very_active = 4
extra_active = 5
sedentary_mf = 1.2
lightly_active_mf = 1.375
moderately_active_mf = 1.55
very_active_mf = 1.725
extra_active_mf = 1.9

# Minimum count of users rated recipes
# Eka recipe la kamit kami 50 users ne rating dila aahe. # 1 recipe to 50 recipes
ratings_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
count_of_users_rated_atleas_50 = 50
filter_recipes = ratings_df['recipe_id'].value_counts() > count_of_users_rated_atleas_50
print("Original #recipes = ", len(recipe_df))
print("filter #recipes = ", len(filter_recipes))
filter_recipes = filter_recipes[filter_recipes].index.tolist()
#
MIN_USER_INTERACTION = 10
MAX_USER_INTERACTION = 350
filter_users = (ratings_df['user_id'].value_counts() > MIN_USER_INTERACTION) & (
        ratings_df['user_id'].value_counts() <= MAX_USER_INTERACTION)
print("filter_users ", len(filter_users))
filter_users = filter_users[filter_users].index.tolist()
filtered_ratings_df = ratings_df[
    (ratings_df['recipe_id'].isin(filter_recipes)) & (ratings_df['user_id'].isin(filter_users))]
print("length of filtered_ratings_df = ", len(filtered_ratings_df))
print("length of unique users in filtered_ratings_df = ", len(filtered_ratings_df.user_id.unique()))
#######

user_df = pd.DataFrame()
# user_df['user_id'] = list(users_with_enough_interactions_df['user_id'].head(10000))
user_df['user_id'] = list(filtered_ratings_df.user_id.unique())[:20000]
print("user_df = ", user_df.shape)
print(user_df.head())
weight_height_df = pd.read_csv(os.path.realpath('../data/original/weight-height.csv'))
frames = [weight_height_df, weight_height_df]
result_wt_ht_df = pd.concat(frames)
weight_height_df = result_wt_ht_df
print(user_df.shape[0])
print(weight_height_df.shape[0])

user_df['Gender'] = weight_height_df['Gender'].to_list()

# print("length of usrs - ", len(users_df), "\n htwt = ", len(weight_height_df))
user_df['Height_inch'] = list(weight_height_df['Height_inch'])
user_df['Weight_lb'] = list(weight_height_df['Weight_lb'])
# convert height to meter
height_mtr = weight_height_df['Height_inch'] * 0.0254
user_df['height_mtr'] = weight_height_df['height_mtr'] = list(height_mtr)
# conver weight to kgs
weight_kgs = weight_height_df['Weight_lb'] * 0.453592
user_df['weight_kgs'] = weight_height_df['weight_kgs'] = list(weight_kgs)
# print(weight_height_df.columns.values)
bmi = weight_height_df['weight_kgs'] / np.power((weight_height_df['height_mtr']), 2)
user_df['BMI'] = list(bmi)

# print("bmi",bmi)
# generate random integer values

# seed random number generator
seed(1)
# generate some integers
age_list = []
activity_list = []
for _ in range(len(user_df)):
    age_list.append(randint(18, 40))
    activity_list.append(randint(1, 5))
user_df['age'] = age_list
user_df['activity'] = activity_list

user_df['BMR'] = user_df.apply(
    lambda row: 66 + (6.3 * row.Weight_lb) + (12.9 * row.Height_inch) - (6.8 * row.age) if (row.Gender == 'Male')
    else 655 + (4.3 * row.Weight_lb) + (4.7 * row.Height_inch) - (4.7 * row.age), axis=1)

user_df['calories_per_day'] = user_df.apply(
    lambda row: row.BMR * sedentary_mf if (row.activity == sedentary)
    else row.BMR * lightly_active_mf if (row.activity == lightly_active)
    else row.BMR * moderately_active_mf if (row.activity == moderately_active)
    else row.BMR * very_active_mf if (row.activity == very_active)
    else row.BMR * extra_active_mf, axis=1)

print("user head = ", user_df.head(5))
print("user tail = ", user_df.tail(5))
user_df.to_csv(r'../data/clean/users_v1.csv', index=False, header=True)
# print("height_mtr = ", height_mtr)
# print("weight_kgs = ", weight_kgs)
# return user_df