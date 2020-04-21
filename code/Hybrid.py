#https://www.kaggle.com/robottums/hybrid-recommender-systems-with-surprise
import surprise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import KNNBasic, SVD, CoClustering, SlopeOne, Reader, Dataset, accuracy
from surprise.model_selection import split
from sklearn.model_selection import train_test_split
from code import Evaluators

class HybridAlgorithm(surprise.AlgoBase):
    def __init__(self, epochs, learning_rate, num_models, knnbasic, svd, coclus, slopeone):
        surprise.AlgoBase.__init__(self)
        self.alpha = np.array([1 / num_models] * num_models)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.knnbasic = knnbasic
        self.svd = svd
        self.coclus = coclus
        self.slopeone = slopeone

    def fit(self, holdout):
        surprise.AlgoBase.fit(self, trainset=holdout.build_full_trainset())
        holdout = holdout.build_full_trainset().build_testset()
        for epoch in range(self.epochs):
            # print("epoch= ", epoch)
            # rmseGradient = []
            # prediction = self.knnbasic.test(traindata)
            # rmseGradient.append(accuracy.rmse(prediction, verbose=False) * self.learning_rate)
            # prediction = self.svd.test(traindata)
            # rmseGradient.append(accuracy.rmse(prediction, verbose=False) * self.learning_rate)
            # prediction = self.coclus.test(traindata)
            # rmseGradient.append(accuracy.rmse(prediction, verbose=False) * self.learning_rate)
            # prediction = self.slopeone.test(traindata)
            # rmseGradient.append(accuracy.rmse(prediction, verbose=False) * self.learning_rate)
            # #convergence check:
            # newalpha = self.alpha - rmseGradient
            # if (newalpha - self.alpha < 0.001).all(): break
            # self.alpha = newalpha

            predictions = np.array([self.knnbasic.test(holdout), self.svd.test(holdout), self.coclus.test(holdout), self.slopeone.test(holdout)])
            rmseGradient = np.array([accuracy.rmse(list(pred), verbose=False) for pred in predictions])
            newalpha = self.alpha - self.learning_rate * rmseGradient
            # convergence check:
            if (newalpha - self.alpha < 0.001).any(): break
            self.alpha = newalpha

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise surprise.PredictionImpossible('User and/or item is unkown.')
        #remove string data from last column
        algoResults = np.array([self.knnbasic.predict(u, i)[:-1], self.svd.predict(u, i)[:-1],
                                self.coclus.predict(u, i)[:-1], self.slopeone.predict(u, i)[:-1]])
        # replace none object type by 0 if true rating is None
        algoResults = np.where(algoResults == None, 0, algoResults)
        return np.sum(np.dot(self.alpha, algoResults))

def ComputeHybrid(recipe_df, train_rating_df, pd, benchmark):
    print("\n###### Compute Hybrid ######")
    df = pd.merge(recipe_df, train_rating_df, on='recipe_id', how='inner')

    # swapping columns
    df = df[['user_id', 'recipe_id', 'rating']]
    df.columns = ['n_users', 'n_items', 'rating']

    # need sklearn for this split ONLY
    rawTrain, rawholdout = train_test_split(df, test_size=0.2)
    # when importing from a DF, you only need to specify the scale of the ratings.
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rawTrain, reader)
    holdout = Dataset.load_from_df(rawholdout, reader)

    # split data into folds.
    kSplit = split.KFold(n_splits=10, shuffle=True)
    sim_options = {'name': 'cosine', 'user_based': False}
    # errors on removing sim_options.
    knnbasic = KNNBasic(k=40, sim_options=sim_options, verbose=False)
    rmseKNN = []
    for trainset, testset in kSplit.split(data):  # iterate through the folds.
        knnbasic.fit(trainset)
        predictionsKNN = knnbasic.test(testset)
        rmseKNN.append(accuracy.rmse(predictionsKNN, verbose=False))  # get root means squared error

    rmseSVD = []
    svd = SVD(n_factors=30,n_epochs=5,biased=True)
    for trainset, testset in kSplit.split(data):  # iterate through the folds.
        svd.fit(trainset)
        predictionsSVD = svd.test(testset)
        rmseSVD.append(accuracy.rmse(predictionsSVD, verbose=False))  # get root means squared error
    #print("predictionsSVD = ", predictionsSVD)
    #print("rmseSVD = ", rmseSVD)
    rmseCo = []
    coclus = CoClustering(n_cltr_u=4,n_cltr_i=4,n_epochs=5)
    for trainset, testset in kSplit.split(data):  # iterate through the folds.
        coclus.fit(trainset)
        predictionsCoClus = coclus.test(testset)
        rmseCo.append(accuracy.rmse(predictionsCoClus, verbose=False))  # get root means squared error

    rmseSlope = []
    slopeone = SlopeOne()
    for trainset, testset in kSplit.split(data):  # iterate through the folds.
        slopeone.fit(trainset)
        predictionsSlope = slopeone.test(testset)
        rmseSlope.append(accuracy.rmse(predictionsSlope, verbose=False))  # get root means squared error

    hybrid = HybridAlgorithm(epochs=10, learning_rate=0.05, num_models=4, knnbasic=knnbasic, svd=svd, coclus=coclus, slopeone=slopeone)
    rmseHybrid = []
    hybrid.fit(holdout)
    for trainset, testset in kSplit.split(data):  # iterate through the folds.
        predictionsHybrid = hybrid.test(testset)
        rmseHybrid.append(accuracy.rmse(predictionsHybrid, verbose=False))  # get root means squared error

    PredArray = [predictionsKNN, predictionsSVD, predictionsCoClus, predictionsSlope, predictionsHybrid]
    DisplayPlot(PredArray, rmseKNN, rmseSVD, rmseCo, rmseSlope, rmseHybrid)
    Evaluators.RunAllEvals(predictionsHybrid, benchmark)
    #precisions, recalls = Evaluators.precision_recall_at_k(predictions=predictionsHybrid)
    # Precision and recall can then be averaged over all users
    #precisionAt10 = sum(prec for prec in precisions.values()) / len(precisions)
    #recallAt10 = sum(rec for rec in recalls.values()) / len(recalls)
    #print("precisionAt10: ", precisionAt10)
    #print("recallAt10: ", recallAt10)

def DisplayPlot(PredArray, rmseKNN, rmseSVD, rmseCo, rmseSlope, rmseHybrid):
    #print("rmseKNN= ", rmseKNN)
    #print("rmseSVD= ", rmseSVD)
    #print("rmseCo= ", rmseCo)
    #print("rmseSlope= ", rmseSlope)
    #print("rmseHybrid= ", rmseHybrid)
    for pred in PredArray:
        plt.plot(rmseKNN, label='knn', color='r')
        plt.plot(rmseSVD, label='svd', color='g')
        plt.plot(rmseCo, label='cluster', color='b')
        plt.plot(rmseSlope, label='slope', color='c')
        plt.plot(rmseHybrid, label='Hybrid', color='y', linestyle='--')

    plt.xlabel('folds (i.e. each computed pred and rmse)')
    plt.ylabel('accuracy (rmse value)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)

    #rmsemetric = pd.DataFrame([rmseKNN, rmseSVD, rmseCo, rmseSlope, rmseHybrid])
    #ax = rmsemetric.transpose().plot(kind='bar', figsize=(15, 8))
    #for p in ax.patches:
    #    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',xytext=(0, 10), textcoords='offset points')
    plt.show()
