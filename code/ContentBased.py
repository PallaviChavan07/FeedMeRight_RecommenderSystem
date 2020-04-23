import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import csv

def _verify_hit_top_n(item_id, recommended_items, topn):
    try:
        index = next(i for i, c in enumerate(recommended_items) if c == item_id)
    except:
        index = -1
    hit = int(index in range(0, topn))
    return hit, index

def ComputeContentBasedFiltering(recipe_df, rating_df, pd):
    print("\n###### Compute Content Based Filtering ######")
    merged_df = pd.merge(recipe_df, rating_df, on='recipe_id', how='inner')
    train_df, test_df = train_test_split(merged_df, test_size=0.20)
    test_indexed_df = test_df.set_index('user_id')

    # Something which we need always hence keeping in common
    # token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' # can be added as parameter in Tfidfvectorizer to remove numbers
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01, max_df=0.80, stop_words='english')
    tfidf_matrix = tf.fit_transform(train_df['ingredients'])
    field_names = tf.get_feature_names()

    get_recipe_features(train_df, field_names)
    user_profile = build_user_profile(train_df, rating_df, 39, pd)
    get_recommendations(merged_df, tfidf_matrix, user_profile, pd, test_indexed_df)


# Get normalized recipe features
# iterate through each recipes ingredient and create it's own features and set it's value if it is present in final list of all unique features.
def get_recipe_features(recipe_df, field_names):
    field_names.insert(0, 'recipe_id')
    print("TF Count = ", len(field_names))#, "\n Feature names =>", field_names)
    with open('../data/codegenerated/recipes_feature_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.00, max_df=1., stop_words='english')
        for i in range(len(recipe_df)):
            try:
                recipe_id = recipe_df.loc[i, "recipe_id"]
                ingredients = recipe_df.loc[i, "ingredients"]
                ingredients = [ingredients]
                tfidf_matrix = tf.fit_transform(ingredients)
                recipe_features = tf.get_feature_names()
                # print("Number of recipe features = ", len(recipe_features))
                # print("Number of common features = ", len(list(set(fieldnames) & set(recipe_features))))
                num_of_common_feaures = len(list(set(field_names) & set(recipe_features)))

                row = dict()
                for field in field_names:
                    if field == 'recipe_id':
                        row[field] = recipe_id
                    elif field in recipe_features:
                        row[field] = 1 / num_of_common_feaures
                    else:
                        row[field] = 0
                writer.writerow(row)
            except:
                continue


def build_user_profile(recipe_df, rating_df, user_id, pd):
    ### Build user profile for user #1
    print("## In build_user_profile =>", user_id)
    allrecipes_feature_matrix_df = pd.read_csv('../data/codegenerated/recipes_feature_matrix.csv')
    df_user_ratings = rating_df[rating_df.user_id == user_id]
    #print("df_user_ratings ==>\n", df_user_ratings)
    df_user_data_with_features = recipe_df.reset_index().merge(df_user_ratings, on='recipe_id')
    df_user_data_with_features['weight'] = df_user_data_with_features['rating_y'] / 5.0
    #print(" df_user_data_with_features = >\n", df_user_data_with_features)
    user_recipes_feature_matrix_df = pd.merge(allrecipes_feature_matrix_df, df_user_data_with_features, how='inner',on='recipe_id')[allrecipes_feature_matrix_df.columns]

    #print("user_recipes_feature_matrix_df ==>\n",user_recipes_feature_matrix_df)
    mylist = []
    ## Get User Profile
    for col in user_recipes_feature_matrix_df.columns:
        if col != 'recipe_id':
            feature_vals = np.array(user_recipes_feature_matrix_df[col])
            mylist.append(feature_vals)

    user_recipe_matrix = np.array(mylist)
    user_rating_vector = np.array(df_user_data_with_features['weight'])
    # get only matrix data but ned to join on recipe id for selected user only.. here 16.. ????????????
    user_profile = np.dot(user_recipe_matrix,user_rating_vector )
    #print("user_profile ===> \n",user_profile)
    return user_profile

def get_recommendations(recipe_df, tfidf_matrix, user_profile, pd, interactions_test_indexed_df):
    #print("user_profile ===> \n",user_profile)
    # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01, max_df=0.80, stop_words='english')
    # tfidf_matrix = tf.fit_transform(recipe_df['ingredients'])
    # C = cosine_similarity(np.atleast_2d(user_profile), allrecipes_feature_matrix_df.loc[:, allrecipes_feature_matrix_df.columns != 'recipe_id'])
    C = linear_kernel(np.atleast_2d(user_profile), tfidf_matrix)
    R = np.argsort(C)[:, ::-1]
    print("R = > \n", R)
    # Select selected indexes with selected columns of dataframe = > recipe_df.loc[[703,698,194], ['recipe_id', 'recipe_name']]
    # Select selected indexes with all columns of dataframe => recipe_df.loc[[703, 698, 194],:]

    recommendations = [i for i in R[0]]
    print("recommendations = >\n", recommendations[:10])
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("recommendations With All details = >\n", recipe_df.loc[recommendations[:10], ['recipe_id', 'recipe_name', 'ingredients']])
    # print(recipe_df.loc[[recommendations], :].head(10))



    recipe_recs_df = recipe_df.loc[recommendations[:10], ['recipe_id']]
    #print(interactions_test_indexed_df.head(20))
    test_user_id = 35215
    interacted_values_testset = interactions_test_indexed_df.loc[test_user_id]
    if type(interacted_values_testset['recipe_id']) == pd.Series:
        person_interacted_items_testset = set(interacted_values_testset['recipe_id'])
    else:
        person_interacted_items_testset = {int(interacted_values_testset['recipe_id'])}
    interacted_items_count_testset = len(person_interacted_items_testset)
    hits_at_5_count = 0
    hits_at_10_count = 0
    # For each item the user has interacted in test set
    for item_id in person_interacted_items_testset:
        # Getting a random sample (100) items the user has not interacted
        # (to represent items that are assumed to be no relevant to the user)
        # non_interacted_items_sample = self.get_not_interacted_items_sample(user_id,
        #                                                                    sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
        #                                                                    seed=item_id % (2 ** 32))

        # Combining the current interacted item with the 100 random items
        #items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

        # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
        if not recipe_recs_df is None:
            #valid_recs = recipe_recs_df['recipe_id'].values
            valid_recs = recipe_recs_df['recipe_id']
        else:
            # this way we can still get person_metrics
            valid_recs = None

        # Verifying if the current interacted item is among the Top-N recommended items
        #print(valid_recs)
        hit_at_5, index_at_5 = _verify_hit_top_n(item_id, valid_recs, 5)
        hits_at_5_count += hit_at_5
        hit_at_10, index_at_10 = _verify_hit_top_n(item_id, valid_recs, 10)
        hits_at_10_count += hit_at_10

    # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
    # when mixed with a set of non-relevant items
    recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
    recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
    print("\ninteracted_items_count_testset: ", interacted_items_count_testset)
    print("hits_at_5_count: ", hits_at_5_count)
    print("recall_at_5: ", recall_at_5)
    print("hits_at_10_count: ", hits_at_10_count)
    print("recall_at_10: ", recall_at_10)
