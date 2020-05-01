########################################## Popularity BASED ##########################################
class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, interactions_full_df, recipe_df=None):
        # Computes the most popular items by grouping recipe_ids with maximum number of ratings and not just maximum ratings
        interactions_full_df = interactions_full_df.loc[interactions_full_df['rating'] > 3]
        self.popularity_df = interactions_full_df.groupby('recipe_id')['rating'].sum().sort_values(ascending=False).reset_index()
        self.recipe_df = recipe_df
        #print(self.popularity_df.head(5))

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, items_to_ignore=[], topn=10, pd=None, newuser_cal_count=1000):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['recipe_id'].isin(items_to_ignore)].sort_values('rating', ascending=False)
        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')

        #get only those recommendations which have cal score < newuser_cal_count
        recommendations_df = recommendations_df.loc[recommendations_df['calories'] <= newuser_cal_count]

        #get 2 recommendations for each dietlabels
        balanced_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('balanced')].head(2)
        highprotein_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('highprotein')].head(2)
        highfiber_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('highfiber')].head(2)
        lowcarb_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('lowcarb')].head(2)
        lowfat_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('lowfat')].head(2)
        lowsodium_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('lowsodium')].head(2)

        #empty old df
        recommendations_df.drop(recommendations_df.index, inplace=True)
        combined_frame = [balanced_df, highprotein_df, highfiber_df, lowcarb_df, lowfat_df, lowsodium_df]
        recommendations_df = pd.concat(combined_frame)
        recommendations_df = recommendations_df.sort_values('rating', ascending=False).head(topn)[['recipe_id', 'recipe_name', 'calories', 'diet_labels']]

        return recommendations_df
