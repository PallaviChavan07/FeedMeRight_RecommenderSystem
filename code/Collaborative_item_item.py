## Source - https://blog.dominodatalab.com/recommender-systems-collaborative-filtering/
## Source - https://www.kaggle.com/robottums/hybrid-recommender-systems-with-surprise

## Collaborative filtering based on item item similarity
from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import split
import Top5_Recipe_Reco_PerUser
import Recipe_Reco_SingleUser

def ComputeCollaborativeFiltering_Item_Item(recipe_df, train_rating_df, pd):
    print("\n###### ComputeCollaborativeFiltering_Item_Item ######")
    df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)

    # split data into folds.
    kSplit = split.KFold(n_splits=5, shuffle=True)

    # compute  similarities between items
    sim_options = {'name': 'cosine', 'user_based': False}

    collabKNN = KNNBasic(k=40, sim_options=sim_options)
    rmseKNN = []

    for trainset, testset in kSplit.split(data):  # iterate through the folds.
        collabKNN.fit(trainset)
        predictionsKNN = collabKNN.test(testset)
        rmseKNN.append(accuracy.rmse(predictionsKNN, verbose=True))  # get root means squared error
    print(rmseKNN)

    #Display Results
    # Top5_Recipe_Reco_PerUser.DisplayResults(predictionsKNN)
    Recipe_Reco_SingleUser.GetSingleUserRecipeReco(df, collabKNN, 39)

