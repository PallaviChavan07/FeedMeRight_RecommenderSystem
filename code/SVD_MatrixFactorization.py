# ## Source - https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b
# ## Notebook Code Reference - https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Movielens%20Recommender%20Metrics.ipynb

from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split, GridSearchCV, cross_validate
from code import Evaluators, Recipe_Reco_SingleUser, Top5_Recipe_Reco_PerUser

def ComputeSVD_MatrixFactorization(recipe_df, train_rating_df, pd, benchmark):
    print("\n###### Compute SVD_MatrixFactorization ######")
    df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)

    # Testing concept of finding out best params:
    #param_grid = {'n_factors': [40, 80, 120, 160], 'n_epochs': [5, 10, 15], 'lr_all': [0.001, 0.003, 0.005, 0.008], 'reg_all': [0.08, 0.1, 0.15]}
    #gs = GridSearchCV(SVD, param_grid, measures=['RMSE'], cv=3, n_jobs=-1)
    #gs.fit(data)
    #algo = gs.best_estimator['rmse']
    #print(gs.best_score['rmse'])
    #print(gs.best_params['rmse'])
    #results = cross_validate(algo, data, measures=['RMSE'], cv=5, n_jobs=-1, verbose=False)
    #print(results)

    algo = SVD(random_state=0)
    algo.fit(trainSet)
    predictions = algo.test(testSet)

    Evaluators.RunAllEvals(predictions, benchmark)

    #Display Results
    #Top5_Recipe_Reco_PerUser.DisplayResults(predictionsKNN)
    #Recipe_Reco_SingleUser.GetSingleUserRecipeReco(df, algo, 39)



