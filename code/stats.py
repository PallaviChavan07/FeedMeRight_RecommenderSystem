import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None, "display.max_columns", None)
recipe_df = pd.read_csv('../data/original/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('../data/original/core-data-train_rating.csv')
print("Total Interactions = ", len(train_rating_df))
pd.value_counts(train_rating_df['rating']).plot.bar()
#plt.plot(train_rating_df['rating'].hist)
#plt.scatter(df['col_name_1'], df['col_name_2'])
#plt.show() #
#creating sample data
# sample_data={'col_name_1':np.random.rand(20),
#       'col_name_2': np.random.rand(20)}
# df= pd.DataFrame(sample_data)
# df.plot(x='col_name_1', y='col_name_2', style='o')

## TODO
## Calculate how many users have interacted min and max times.
## how many ratings are there for each user
users_interactions_count_df = train_rating_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
maxinteracted_users = users_interactions_count_df.sort_values(ascending=False).head(10010)
print("maxinteracted_users length = ", maxinteracted_users)
#print(maxinteracted_users)
print("max = ",max(users_interactions_count_df))
print("min = ",min(users_interactions_count_df))

# check count of number of user who have rated min 15 and max 150 recipes
MIN_USERS_INTERACTIONS = 15
MAX_USERS_INTERACTIONS = 120
users_with_enough_interactions_df = users_interactions_count_df[(users_interactions_count_df >= MIN_USERS_INTERACTIONS) & (users_interactions_count_df < MAX_USERS_INTERACTIONS)].reset_index()[['user_id']]
print("users_with_enough_interactions_df = ", len(users_with_enough_interactions_df))