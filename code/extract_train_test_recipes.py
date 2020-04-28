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
train_ratings_df = pd.read_csv('C:/Users/jpall/D/thesis/data_gathering/Kaggle_recipe_huge_26th March/smalldataset/core-data-train_rating.csv')
test_ratings_df = pd.read_csv('C:/Users/jpall/D/thesis/data_gathering/Kaggle_recipe_huge_26th March/smalldataset/core-data-test_rating.csv')
core_recipes_df = pd.read_csv('C:/Users/jpall/D/thesis/data_gathering/Kaggle_recipe_huge_26th March/smalldataset/core-data_recipe.csv')
recipe_df = pd.read_csv('../data/original/export_rated_recipes_set.csv')
ratings_df = pd.read_csv('../data/original/core-data-train_rating.csv')
# print("train unique = ",len(train_recipe_ids))
# print("test unique = ",len(test_recipe_ids))
# print("total set unique = ",len(total_set_recipe_ids))


# In[8]:
def get_rated_recipes():
    train_recipe_ids = train_ratings_df.recipe_id.unique()
    test_recipe_ids = test_ratings_df.recipe_id.unique()
    total_set_recipe_ids = core_recipes_df.recipe_id.unique()
    # Combine train and test sets and get unique recipe_ids from them
    train_test_unique_ids = list(set(train_recipe_ids) | set(test_recipe_ids))
    print(len(train_test_unique_ids))
    rated_recipes_list = []
    rated_nonexist_list = []
    for train_test_id in train_test_unique_ids:
        if train_test_id in total_set_recipe_ids:
            rated_recipes_list.append(train_test_id)
        else:
            rated_nonexist_list.append(train_test_id)

    print("rated recipes = ", len(rated_recipes_list))
    print("rated nonexist recipes = ", len(rated_nonexist_list))
    rated_recipes_set = core_recipes_df.loc[core_recipes_df['recipe_id'].isin(rated_recipes_list)]
    rated_recipes_set.head()
    # print(len(rated_recipes_set))
    rated_recipes_set.to_csv( r'../data/original/export_rated_recipes_set.csv', index=False, header=True)


def get_unique_users():

    print("user-recipe-rating interactions = ", len(ratings_df))
    dup_users_df = pd.DataFrame()
    users_df = pd.DataFrame()
    dup_users_df['user_id'] = ratings_df['user_id'].to_numpy()
    print("Number of users = ", len(dup_users_df))
    users_df['user_id'] = ratings_df['user_id'].unique()
    print("Number of unique users = ", len(users_df ))
    return users_df


def bmi_calculations():
    users_interactions_count_df = train_ratings_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
    MIN_USERS_INTERACTIONS = 15
    MAX_USERS_INTERACTIONS = 120
    users_with_enough_interactions_df = users_interactions_count_df[ (users_interactions_count_df >= MIN_USERS_INTERACTIONS) &
                                        (users_interactions_count_df < MAX_USERS_INTERACTIONS)].reset_index()[['user_id']]
    user_df = pd.DataFrame()
    user_df['user_id'] = list(users_with_enough_interactions_df['user_id'].head(10000))
    #print("user_df = ", user_df.head(4))
    print(user_df.head())
    weight_height_df = pd.read_csv('../data/original/weight-height.csv')
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


    user_df.to_csv(r'../data/original/users.csv', index=False, header=True)
    # print("height_mtr = ", height_mtr)
    # print("weight_kgs = ", weight_kgs)



def clean_recipes_for_calories():
    print(recipe_df.columns.values)
    nutritions_lst = recipe_df['nutritions'].tolist()
    calories_list = []
    for nut in nutritions_lst:
        nut = ast.literal_eval(nut)
        #print("nut['calories'] == ",nut['calories']['amount'], nut['calories']['unit'])
        calories_list.append(nut['calories']['amount'])
    recipe_df['calories'] = calories_list
    recipe_df.to_csv(r'../data/original/recipes_with_calories.csv', index=False, header=True)
if __name__ == '__main__':
    #clean_recipes_for_calories()
    #users_df = get_unique_users()
    bmi_calculations()


