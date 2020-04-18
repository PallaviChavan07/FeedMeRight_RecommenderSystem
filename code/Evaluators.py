from surprise import accuracy

def MAE(predictions):
    # get mean absolute error
    return accuracy.mae(predictions, verbose=False)

def RMSE(predictions):
    # get root means squared error
    return accuracy.rmse(predictions, verbose=False)