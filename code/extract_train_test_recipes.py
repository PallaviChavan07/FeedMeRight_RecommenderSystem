### TODO
### Check how many ratings are given by each user.. and select maybe only those users who have given approximately similar number of recipes.

#Extract train test recipes from toalt set of recipes. Because we want to work on only recipes which are rated by user. 
import pandas as pd
import json
import ast
import numpy as np
from random import seed
from random import randint
pd.set_option("display.max_rows", None, "display.max_columns", None)
train_ratings_df = pd.read_csv('../data/original/core-data-train_rating.csv')
test_ratings_df = pd.read_csv('../data/original/core-data-test_rating.csv')
train_test_ratings_df = pd.concat([train_ratings_df, test_ratings_df], ignore_index=True)
core_recipes_df = pd.read_csv('../data/original/core-data_recipe.csv')

def get_rated_recipes(core_recipes_df):
    # Get all recipe ids from all interactions
    interaction_recipe_ids = train_test_ratings_df.recipe_id.unique()
    # Get all unique recipes from core recipes set
    all_unique_recipe_ids = core_recipes_df.recipe_id.unique()
    # Common recipes from interacted recipes and actual recipe data = all rated recipes
    rated_recipe_ids = list(set(interaction_recipe_ids) & set(all_unique_recipe_ids))
    rated_recipes_df = core_recipes_df.loc[core_recipes_df['recipe_id'].isin(rated_recipe_ids)]
    valid_interactions_df = train_test_ratings_df.loc[train_test_ratings_df['recipe_id'].isin(rated_recipe_ids)]

    print("Actual number of recipes = ", len(core_recipes_df))
    print("all unique recipes = ", len(all_unique_recipe_ids))
    print("rated recipes = ", len(rated_recipes_df))
    print("interacted unique recipes = ", len(interaction_recipe_ids))
    print("Number of interactions = ", len(train_test_ratings_df))
    print("valid interacted recipes = ", len(valid_interactions_df))
    return rated_recipes_df, valid_interactions_df

# def get_unique_users():
#
#     print("user-recipe-rating interactions = ", len(ratings_df))
#     dup_users_df = pd.DataFrame()
#     users_df = pd.DataFrame()
#     dup_users_df['user_id'] = ratings_df['user_id'].to_numpy()
#     print("Number of users = ", len(dup_users_df))
#     users_df['user_id'] = ratings_df['user_id'].unique()
#     print("Number of unique users = ", len(users_df ))
#     return users_df


def user_data_generation(rated_recie_df, ratings_df):
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
    users_interactions_count_df = ratings_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
    MIN_USERS_INTERACTIONS = 20
    # MAX_USERS_INTERACTIONS = 120
    # users_with_enough_interactions_df = users_interactions_count_df[ (users_interactions_count_df >= MIN_USERS_INTERACTIONS) &
    #                                     (users_interactions_count_df < MAX_USERS_INTERACTIONS)].reset_index()[['user_id']]
    users_with_enough_interactions_df = users_interactions_count_df[(users_interactions_count_df >= MIN_USERS_INTERACTIONS)].reset_index()[['user_id']]
    user_df = pd.DataFrame()
    #user_df['user_id'] = list(users_with_enough_interactions_df['user_id'].head(10000))
    user_df['user_id'] = list(users_with_enough_interactions_df['user_id'].tail(10000))
    #print("user_df = ", user_df.head(4))
    print(user_df.head())
    weight_height_df = pd.read_csv('../data/original/weight-height.csv')
    print(user_df.shape[0])
    print(weight_height_df.shape[0])
    user_df['Gender'] = weight_height_df['Gender'].to_list()

    #print("length of usrs - ", len(users_df), "\n htwt = ", len(weight_height_df))
    user_df['Height_inch'] = list(weight_height_df['Height_inch'])
    user_df['Weight_lb'] = list(weight_height_df['Weight_lb'])
    # convert height to meter
    height_mtr =  weight_height_df['Height_inch']*0.0254
    user_df['height_mtr'] = weight_height_df['height_mtr'] = list(height_mtr)
    # conver weight to kgs
    weight_kgs = weight_height_df['Weight_lb']*0.453592
    user_df['weight_kgs'] = weight_height_df['weight_kgs'] = list(weight_kgs)
    #print(weight_height_df.columns.values)
    bmi = weight_height_df['weight_kgs'] / np.power((weight_height_df['height_mtr']),2)
    user_df['BMI'] = list(bmi)

    #print("bmi",bmi)
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
    user_df['activity'] =activity_list

    user_df['BMR'] = user_df.apply(
        lambda row: 66 + (6.3 * row.Weight_lb) + (12.9 * row.Height_inch) - (6.8 * row.age) if (row.Gender == 'Male')
        else 655 + (4.3 * row.Weight_lb) + (4.7 * row.Height_inch) - (4.7 * row.age), axis = 1)

    user_df['calories_per_day'] = user_df.apply(
        lambda row: row.BMR * sedentary_mf  if (row.activity == sedentary)
        else row.BMR * lightly_active_mf if (row.activity == lightly_active)
        else row.BMR * moderately_active_mf  if (row.activity == moderately_active)
        else row.BMR * very_active_mf if (row.activity == very_active)
        else row.BMR * extra_active_mf, axis=1)

    #user_df.to_csv(r'../data/original/clean/users.csv', index=False, header=True)
    # print("height_mtr = ", height_mtr)
    # print("weight_kgs = ", weight_kgs)
    return user_df


def drop_recipes_with_no_calories():
    core_recipes_df = pd.read_csv('../data/original/core-data_recipe.csv')
    nutritions_lst = core_recipes_df['nutritions'].tolist()
    calories_list = []
    for nut in nutritions_lst:
        nut = ast.literal_eval(nut)
        calories_list.append(nut['calories']['amount'])
    core_recipes_df['calories'] = calories_list
    print("bfore removing 0 calories ", len(core_recipes_df))
    core_recipes_df = core_recipes_df[core_recipes_df['calories'] != 0]
    print("After removing 0 calories ", len(core_recipes_df))
    return core_recipes_df
    #recipe_df.to_csv(r'../data/original/recipes_with_calories.csv', index=False, header=True)
if __name__ == '__main__':
    core_recipes_df = drop_recipes_with_no_calories()
    rated_recie_df, valid_interactions_df = get_rated_recipes(core_recipes_df)
    user_df = user_data_generation(rated_recie_df, valid_interactions_df)
    rated_recie_df.to_csv(r'../data/clean/recipes.csv', index=False, header=True)
    valid_interactions_df.to_csv(r'../data/clean/ratings.csv', index=False, header=True)
    user_df.to_csv(r'../data/clean/users.csv', index=False, header=True)

