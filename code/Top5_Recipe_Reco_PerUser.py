from collections import defaultdict

# helper method to get only top 5 recipe recommendation for each user.
def get_top5_recommendations(predictions, topN=5):
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))

    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recs[uid] = user_ratings[:topN]

    return top_recs

def DisplayResults(predictionsKNN):
    print("\n###### Recipe Reco Top 5 Per User ######")
    # helper method to create a dictionary that maps each recipe is to it's name
    top_5 = get_top5_recommendations(predictionsKNN)
    # Print the recommended items for each user
    for uid, user_ratings in top_5.items():
        if (uid == 16):
            print(uid, [iid for (iid, _) in user_ratings])

