import pandas as pd
import os
#data
recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
rating_df = pd.read_csv(os.path.realpath('../data/clean/ratings.csv'))
user_df = pd.read_csv(os.path.realpath('../data/clean/users.csv'))

#user_df = user_df.head(1000)
# valid_users_interaction_df is a subset of rating_df
valid_users_interaction_df = pd.merge(rating_df, user_df, on='user_id', how='inner')
merged_df = pd.merge(recipe_df, valid_users_interaction_df, on='recipe_id', how='inner')
# get unique recipes from merged df
unique_valid_recipes = merged_df.recipe_id.unique()

print("Original recipe df length = ", len(recipe_df))
recipe_df = recipe_df[recipe_df['recipe_id'].isin(unique_valid_recipes)]
print("Filtered recipes df length = ", len(recipe_df))
interactions_df = merged_df[['user_id', 'recipe_id', 'rating']]

