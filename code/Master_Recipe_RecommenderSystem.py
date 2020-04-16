import pandas as pd
import Collaborative_item_item as ctt
import ContentBased as cb

#Read Data
recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')
test_rating_df = pd.read_csv('../data/small_10k/core-data-test_rating.csv')

cb.ComputeContentBasedFiltering(recipe_df, train_rating_df, pd)
ctt.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd)
