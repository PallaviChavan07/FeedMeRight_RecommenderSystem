import pandas as pd
import code.Normal_Predictor as np
import code.Collaborative_item_item as cii
import code.Collaborative_user_user as cuu
import code.ContentBased as cb
import code.SVD_MatrixFactorization as svd_mf
import code.SVDplusplus as svdpp
import code.SlopeOne as slopeone
import code.CoClustering as coclust
import code.Hybrid as hyd

#Read Data
recipe_df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
train_rating_df = pd.read_csv('../data/small_10k/core-data-train_rating.csv')
test_rating_df = pd.read_csv('../data/small_10k/core-data-test_rating.csv')

#stats after duplicates (if any) removal
train_rating_df.drop_duplicates(inplace=True)
print('we have',train_rating_df.shape[0], 'ratings')
print('the number of unique users we have is:', len(train_rating_df.user_id.unique()))
print('the number of unique recipes we have is:', len(train_rating_df.recipe_id.unique()))
print("The median user rated %d books."%train_rating_df.user_id.value_counts().median())
print('The max rating is: %d'%train_rating_df.rating.max(),"and the min rating is: %d"%train_rating_df.rating.min())
#print(train_rating_df.head())

benchmark = []
#cb.ComputeContentBasedFiltering(recipe_df, train_rating_df, pd)
# np.Normalpredictor(recipe_df, train_rating_df, pd, benchmark)
# cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd, benchmark, knnmeans=False)
# cii.ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd, benchmark, knnmeans=True)
# cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd, benchmark, knnmeans=False)
# cuu.ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd, benchmark, knnmeans=True)
# svd_mf.ComputeSVD_MatrixFactorization(recipe_df, train_rating_df, pd, benchmark)
# svdpp.SVDplusplus(recipe_df, train_rating_df, pd, benchmark)
# slopeone.Slopeone(recipe_df, train_rating_df, pd, benchmark)
# coclust.Coclustering(recipe_df, train_rating_df, pd, benchmark)

pd.set_option("display.max_rows", None, "display.max_columns", None)
results = pd.DataFrame.from_records(benchmark, exclude=['MSE', 'FCP'], columns=['RMSE', 'MAE', 'MSE', 'FCP', 'PrecisionAt10', 'RecallAt10'],
                                    index=['NormalPredictor', 'KNNBasic_Item_Item', 'KNNWithMeans_Item_Item', 'KNNBasic_User_User', 'KNNWithMeans_Item_Item', 'SVD', 'SVD++', 'SlopeOne', 'CoClustering'])
print(results)


#hybrid
hyd.ComputeHybrid(recipe_df, train_rating_df, pd)
