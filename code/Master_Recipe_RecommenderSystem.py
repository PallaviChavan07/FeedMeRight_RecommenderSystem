import pandas as pd
import code.Collaborative_item_item as cii
import code.Collaborative_user_user as cuu
import code.ContentBased as cb
import code.SVD_MatrixFactorization as svd_mf
import code.SVDplusplus as svdpp

#Read Data
recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')
test_rating_df = pd.read_csv('../data/small_10k/core-data-test_rating.csv')

#cb.ComputeContentBasedFiltering(recipe_df, train_rating_df, pd)
cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd)
cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd)
svd_mf.ComputeSVD_MatrixFactorization(recipe_df, train_rating_df, pd)
svdpp.SVDplusplus(recipe_df, train_rating_df, pd)
