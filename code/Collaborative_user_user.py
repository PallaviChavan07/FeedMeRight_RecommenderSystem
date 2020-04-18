## Source - https://blog.dominodatalab.com/recommender-systems-collaborative-filtering/
## Source - https://www.kaggle.com/robottums/hybrid-recommender-systems-with-surprise

## Collaborative filtering based on item item similarity
from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise.model_selection import split
from code import Evaluators, Recipe_Reco_SingleUser, Top5_Recipe_Reco_PerUser

def ComputeCollaborativeFiltering_User_User(recipe_df, train_rating_df, pd):
    print("\n###### ComputeCollaborativeFiltering_User_User ######")
    df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)

    # compute  similarities between items
    sim_options = {'name': 'cosine', 'user_based': True}

    #Method 1:
    algo = KNNBasic(k=40, sim_options=sim_options)
    algo.fit(trainSet)
    predictions = algo.test(testSet)

    print("RMSE: ", Evaluators.RMSE(predictions))
    print("MAE: ", Evaluators.MAE(predictions))

    #Display Results
    #Top5_Recipe_Reco_PerUser.DisplayResults(predictionsKNN)
    #Recipe_Reco_SingleUser.GetSingleUserRecipeReco(df, algo, 39)

