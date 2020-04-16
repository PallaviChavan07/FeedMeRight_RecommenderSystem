import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import csv

def ComputeContentBasedFiltering(recipe_df, rating_df, pd):
    print("\n###### Compute Content Based Filtering ######")
    # Something which we need always hence keeping in common
    # token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' # can be added as parameter in Tfidfvectorizer to remove numbers
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01, max_df=0.80, stop_words='english')
    tfidf_matrix = tf.fit_transform(recipe_df['ingredients'])
    field_names = tf.get_feature_names()

    get_recipe_features(recipe_df, field_names)
    user_profile = build_user_profile(recipe_df, rating_df, 39, pd)
    get_recommendations(recipe_df, tfidf_matrix, user_profile)


# Get normalized recipe features
# iterate through each recipes ingredient and create it's own features and set it's value if it is present in final list of all unique features.
def get_recipe_features(recipe_df, field_names):
    field_names.insert(0, 'recipe_id')
    print("TF Count = ", len(field_names))#, "\n Feature names =>", field_names)
    #with open('allrecipes_unique_feature.csv', 'w', newline='') as csvfile:
    with open('../data/codegenerated/recipes_feature_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.00, max_df=1., stop_words='english')
        for i in range(len(recipe_df)):
            recipe_id = recipe_df.loc[i, "recipe_id"]
            ingredients = recipe_df.loc[i, "ingredients"]
            ingredients = [ingredients]
            tfidf_matrix = tf.fit_transform(ingredients)
            recipe_features = tf.get_feature_names()
            #print("Number of recipe features = ", len(recipe_features))
            #print("Number of common features = ", len(list(set(fieldnames) & set(recipe_features))))
            num_of_common_feaures = len(list(set(field_names) & set(recipe_features)))

            row = dict()
            for field in field_names:

                if field == 'recipe_id':
                    row[field] = recipe_id
                elif field in recipe_features:
                    row[field] = 1/num_of_common_feaures
                else:
                    row[field] = 0
            writer.writerow(row)

def build_user_profile(recipe_df, rating_df, user_id, pd):
    ### Build user profile for user #1
    print("## In build_user_profile =>", user_id)
    allrecipes_feature_matrix_df = pd.read_csv('../data/codegenerated/recipes_feature_matrix.csv')
    df_user_ratings = rating_df[rating_df.user_id == user_id]
    #print("df_user_ratings ==>\n", df_user_ratings)
    df_user_data_with_features = recipe_df.reset_index().merge(df_user_ratings, on='recipe_id')
    df_user_data_with_features['weight'] = df_user_data_with_features['rating'] / 5.0
    #print(" df_user_data_with_features = >\n", df_user_data_with_features)
    user_recipes_feature_matrix_df = pd.merge(allrecipes_feature_matrix_df, df_user_data_with_features, how='inner',on='recipe_id')[allrecipes_feature_matrix_df.columns]

    #print("user_recipes_feature_matrix_df ==>\n",user_recipes_feature_matrix_df)
    list = []
    ## Get User Profile
    for col in user_recipes_feature_matrix_df.columns:
        if(col != 'recipe_id'):
            feature_vals = np.array(user_recipes_feature_matrix_df[col])
            list.append(feature_vals)

    user_recipe_matrix = np.array(list)
    user_rating_vector = np.array(df_user_data_with_features['weight'])
    # get only matrix data but ned to join on recipe id for selected user only.. here 16.. ????????????
    user_profile = np.dot(user_recipe_matrix,user_rating_vector )
    #print("user_profile ===> \n",user_profile)
    return user_profile

def get_recommendations(recipe_df, tfidf_matrix, user_profile):
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

    # print("### Printing Recommendations -- ")
    print("recommendations With All details = >\n", recipe_df.loc[recommendations[:10], ['recipe_id', 'recipe_name', 'ingredients']])
    # print(recipe_df.loc[[recommendations], :].head(10))



