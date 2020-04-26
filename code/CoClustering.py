# ## Source - https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b
# ## Notebook Code Reference - https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Movielens%20Recommender%20Metrics.ipynb

from surprise import CoClustering
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from code import Evaluators, Recipe_Reco_SingleUser, Top5_Recipe_Reco_PerUser

def Coclustering(recipe_df, train_rating_df, pd, benchmark):
    print("\n###### Compute CoClustering ######")
    df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)

    algo = CoClustering()
    algo.fit(trainSet)
    predictions = algo.test(testSet)

    Evaluators.RunAllEvals(predictions, benchmark)

    #Display Results
    #Top5_Recipe_Reco_PerUser.DisplayResults(predictionsKNN)
    #Recipe_Reco_SingleUser.GetSingleUserRecipeReco(df, algo, 39)



