########################################## HYBRID FILTERING BASED ##########################################
class HybridRecommender:
    MODEL_NAME = 'Hybrid'
    CB_WEIGHT = 0.3
    CF_WEIGHT = 0.7

    def __init__(self, cb_rec_model, cf_rec_model, recipe_df, user_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.recipe_df = recipe_df
        self.user_df = user_df

    def get_model_name(self):
        return self.MODEL_NAME

    def get_recommendation_for_user_calorie_count(self, cal_rec_df, user_id):
        # print("Hybrid: Before calories filter = ", recommendations_df.shape)
        # get calories required for user
        user_calories_per_day = self.user_df.loc[self.user_df['user_id'] == user_id]['calories_per_day'].values
        # print("Hybrid: user calories per day", user_calories_per_day, type(user_calories_per_day), user_calories_per_day[0])
        # divide calories into 1/3rd part
        user_calories = user_calories_per_day[0] / 3
        # consider only those recipes which have calories less than required calories for that user
        cal_rec_df = cal_rec_df[cal_rec_df['calories'] <= user_calories]
        # print("Hybrid: After calories filter = ", recommendations_df.shape)
        return cal_rec_df

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Getting the top-1000 Content-based filtering recommendations
        try:
            cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        except:
            return None

        # Getting the top-1000 Collaborative filtering recommendations
        try:
            cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        except:
            return None

        # Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df, how='outer', left_on='recipe_id', right_on='recipe_id').fillna(0.0)
        #print(recs_df.head(5))

        # Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrength'] = (recs_df['recStrengthCB'] * self.CB_WEIGHT) + (recs_df['recStrengthCF'] * self.CF_WEIGHT)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrength', ascending=False).head(topn)
        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'calories', 'diet_labels']]
        recommendations_df = self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)
        return recommendations_df