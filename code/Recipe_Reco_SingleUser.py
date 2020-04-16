import numpy as np

## Different way to get recommendation for single user
## Don't want to recommend same recipes that they already have tried

def GetSingleUserRecipeReco(df, collabKNN, userId):
    print("\n###### Recipe Reco For Single User ######")
    # Get a list of all recipe ids
    unique_recipe_ids = df['recipe_id'].unique()
    # Get a list of all recipe ids that has beenn rated by userId
    recipe_ids_ratedby_user = df.loc[df['user_id'] == userId, 'recipe_id']
    # remove the recipe_ids that user -- (with userId) has rated from the list of all recipe ids.
    recipeids_to_pred = np.setdiff1d(unique_recipe_ids, recipe_ids_ratedby_user)
    print("unique_recipe_ids = ", len(unique_recipe_ids))
    print("recipe_ids_ratedby_user = ", len(recipe_ids_ratedby_user))
    print("recipeids_to_pred = ", len(recipeids_to_pred))

    testset = [[userId, iid, 1.] for iid in recipeids_to_pred]
    predictions = collabKNN.test(testset)
    print(predictions[0])
    pred_ratings = np.array([pred.est for pred in predictions])

    # find the index of the maximum predicted rating
    i_max = pred_ratings.argmax()
    # use this to find the corresponding recipeid to recommend
    recipe_id = recipeids_to_pred[i_max]
    print("top recipe for user ", userId, "is with recipe id = ", recipe_id, " with predicted rating = ", pred_ratings[i_max])

