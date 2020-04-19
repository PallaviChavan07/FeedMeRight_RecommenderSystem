import surprise
from surprise import accuracy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#raw=pd.read_csv('../data/small_10k/core-data-train_rating.csv')
raw=pd.read_csv('../data/original/ratings.csv')
raw.drop_duplicates(inplace=True)
print('we have',raw.shape[0], 'ratings')
print('the number of unique users we have is:', len(raw.user_id.unique()))
print('the number of unique books we have is:', len(raw.book_id.unique()))
print("The median user rated %d books."%raw.user_id.value_counts().median())
print('The max rating is: %d'%raw.rating.max(),"the min rating is: %d"%raw.rating.min())
raw.head()

#swapping columns
raw=raw[['user_id','book_id','rating']]
raw.columns = ['n_users','n_items','rating']

rawTrain,rawholdout = train_test_split(raw, test_size=0.2)
# when importing from a DF, you only need to specify the scale of the ratings.
reader = surprise.Reader(rating_scale=(1,5))
#into surprise:
data = surprise.Dataset.load_from_df(rawTrain,reader)
holdout = surprise.Dataset.load_from_df(rawholdout,reader)

# split data into folds.
kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True)
sim_options = {'name': 'cosine', 'user_based': False}
collabKNN = surprise.KNNBasic(k=40,sim_options=sim_options,verbose=False) #try removing sim_options. You'll find memory errors.
rmseKNN = []
for trainset, testset in kSplit.split(data): #iterate through the folds.
    collabKNN.fit(trainset)
    predictionsKNN = collabKNN.test(testset)
    rmseKNN.append(accuracy.rmse(predictionsKNN,verbose=False))#get root means squared error

rmseSVD = []
funkSVD = surprise.prediction_algorithms.matrix_factorization.SVD(n_factors=30,n_epochs=10,biased=True)
min_error = 1
for trainset, testset in kSplit.split(data): #iterate through the folds.
    funkSVD.fit(trainset)
    predictionsSVD = funkSVD.test(testset)
    rmseSVD.append(accuracy.rmse(predictionsSVD,verbose=False))#get root means squared error

rmseCo = []
coClus = surprise.prediction_algorithms.co_clustering.CoClustering(n_cltr_u=4,n_cltr_i=4,n_epochs=25)
for trainset, testset in kSplit.split(data): #iterate through the folds.
    coClus.fit(trainset)
    predictionsCoClus = coClus.test(testset)
    rmseCo.append(accuracy.rmse(predictionsCoClus,verbose=False))#get root means squared error

rmseSlope = []
slopeOne = surprise.prediction_algorithms.slope_one.SlopeOne()
for trainset, testset in kSplit.split(data): #iterate through the folds.
    slopeOne.fit(trainset)
    predictionsSlope = slopeOne.test(testset)
    rmseSlope.append(accuracy.rmse(predictionsSlope,verbose=False))#get root means squared error



class Hybrid(surprise.AlgoBase):
    def __init__(self, epochs, learning_rate, num_models):
        self.alpha = np.array([1 / len(num_models)] * len(num_models))
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, holdout):
        holdout = holdout.build_full_trainset().build_testset()
        for epoch in range(self.epochs):
            predictions = np.array([collabKNN.test(holdout), funkSVD.test(holdout), coClus.test(holdout), slopeOne.test(holdout)])
            print(predictions)
            rmseGradient = [surprise.accuracy.rmse(prediction) for prediction in predictions]
            newalpha = self.alpha - self.learning_rate * rmseGradient
            # convergence check:
            if newalpha - self.alpha < 0.001:
                break
            self.alpha = newalpha

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise surprise.PredictionImpossible('User and/or item is unkown.')
        algoResults = np.array([collabKNN.predict(u, i), funkSVD.predict(u, i), coClus.predict(u, i), slopeOne.predict(u, i)])
        return np.sum(np.dot(self.alpha, algoResults))


hybrid = Hybrid(epochs=10,learning_rate=0.05,num_models={1,2,3,4})
hybrid.fit(holdout)
rmseHyb = []
for trainset, testset in kSplit.split(data): #iterate through the folds.
    predhybrid = hybrid.test(testset)
    rmseHyb.append(accuracy.rmse(predhybrid))

compiledPredictions = [predictionsKNN, predictionsSVD, predictionsCoClus, predictionsSlope, predhybrid]
for prediction in compiledPredictions:
    modelPrediction = plt.plot(rmseKNN,label='knn')
    modelPrediction = plt.plot(rmseSVD,label='svd')
    modelPrediction = plt.plot(rmseCo,label='cluster')
    modelPrediction = plt.plot(rmseSlope,label='slope')
    modelPrediction = plt.plot(rmseHyb, label='Hybrid')
    modelPrediction = plt.xlabel('folds')
    modelPrediction = plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
