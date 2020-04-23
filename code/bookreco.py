#https://github.com/Divyanshu169/IT556_Worthless_without_coffee_DA-IICT_Final_Project
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split

class hybrid(object):
    def __init__(self, user_id, ratings):
        self.user_id = user_id
        self.md = pd.read_csv('../data/original/CustomData/FinalData.csv')
        self.ratings = ratings
        #print("Existing rated items for by user: \n", ratings[(ratings['user_id'] == user_id)][['user_id', 'book_id', 'rating']])

        #self.popularity_rating = self.popularity(self.md)
        self.collaborative_rating = self.collaborative(self.ratings, self.user_id)
        self.content_rating = self.content_based(self.md, self.ratings, self.user_id)
        self.final_hybrid(self.md)

    # Popularity#
    # def popularity(self, md):
    #     print("#################### popularity ####################")
    #     fd = pd.read_csv('../data/original/CustomData/AverageRatings.csv')
    #     fd1 = pd.read_csv('../data/original/CustomData/RatingsCount.csv')
    #
    #     fd[fd['rating'].notnull()]['rating'] = fd[fd['rating'].notnull()]['rating'].astype('float')
    #     vote_averages = fd[fd['rating'].notnull()]['rating']
    #     C = vote_averages.mean()
    #
    #     fd1[fd1['rating'].notnull()]['rating'] = fd1[fd1['rating'].notnull()]['rating'].astype('float')
    #     vote_counts = fd1[fd1['rating'].notnull()]['rating']
    #     m = len(vote_counts)
    #
    #     md['ratings_count'] = fd1['rating']
    #     md['average_rating'] = fd['rating']
    #
    #     qualified = md[(md['ratings_count'].notnull())][
    #         ['book_id', 'title', 'authors', 'ratings_count', 'average_rating']]
    #
    #     qualified['ratings_count'] = qualified['ratings_count'].astype('float')
    #
    #     qualified['average_rating'] = qualified['average_rating'].astype('float')
    #
    #     qualified.shape
    #
    #     def weighted_rating(x):
    #         v = x['ratings_count']
    #         R = x['average_rating']
    #         return (v / (v + m) * R) + (m / (m + v) * C)
    #
    #     qualified['popularity_rating'] = qualified.apply(weighted_rating, axis=1)
    #     pop = qualified[['book_id', 'popularity_rating']]
    #
    #     print(pop.columns)
    #     return pop

    ### Collaborative ##
    def collaborative(self, ratings, user_id):
        print("#################### collaborative ####################")
        reader = Reader()
        data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
        trainSet, testSet = train_test_split(data, test_size=.2, random_state=0)

        ## Training the data ##
        svd = SVD()
        cross_validate(svd, data, measures=['RMSE', 'MAE'])
        algo = SVD()
        algo.fit(trainSet)

        ## Testing the data ##
        predictions = algo.test(testSet)

        count = 0
        for uid, iid, true_r, est, _ in predictions:
            if uid == user_id:
                count = count + 1
                ratings.loc[len(ratings) + 1] = [uid, iid, est]

        cf = ratings[(ratings['user_id'] == user_id)][['book_id', 'rating']]
        cf.columns = ['book_id', 'cf_rating']

        print(cf.columns)
        print("# of content ratings computed:", len(cf['cf_rating']))
        return cf

    ##### CONTENT ######
    def content_based(self, md, ratings, user_id):
        print("#################### content_based ####################")
        md['book_id'] = md['book_id'].astype('int')
        ratings['book_id'] = ratings['book_id'].astype('int')
        ratings['user_id'] = ratings['user_id'].astype('int')
        ratings['rating'] = ratings['rating'].astype('int')
        md['authors'] = md['authors'].str.replace(' ', '')
        md['authors'] = md['authors'].str.lower()
        md['authors'] = md['authors'].str.replace(',', ' ')

        # print(md.head())

        md['authors'] = md['authors'].apply(lambda x: [x, x])
        # print(md['authors'])

        md['Genres'] = md['Genres'].str.split(';')
        # print(md['Genres'])

        md['soup'] = md['authors'] + md['Genres']
        # print(md['soup'])

        md['soup'] = md['soup'].str.join(' ')

        # md['soup'].fillna({})
        # print(md['soup'])

        count = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
        count_matrix = count.fit_transform(md['soup'])
        #print(count_matrix.shape)
        # print np.array(count.get_feature_names())
        # print(count_matrix.shape)

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        def build_user_profiles():
            user_profiles = np.zeros((60001, 999))
            # taking only the first 100000 ratings to build user_profile
            for i in range(0, 100000):
                try:
                    u = ratings.iloc[i]['user_id']
                    b = ratings.iloc[i]['book_id']
                    user_profiles[u][b - 1] = ratings.iloc[i]['rating']
                except:
                    continue
            return user_profiles

        user_profiles = build_user_profiles()

        def _get_similar_items_to_user_profile(person_id):
            # Computes the cosine similarity between the user profile and all item profiles
            user_ratings = np.empty((999, 1))
            cnt = 0
            for i in range(0, 998):
                book_sim = cosine_sim[i]
                user_sim = user_profiles[person_id]
                user_ratings[i] = (book_sim.dot(user_sim)) / sum(cosine_sim[i])
            maxval = max(user_ratings)

            for i in range(0, 998):
                user_ratings[i] = ((user_ratings[i] * 5.0) / (maxval))

                if (user_ratings[i] > 3):
                    cnt += 1

            return user_ratings

        content_ratings = _get_similar_items_to_user_profile(user_id)

        num = md[['book_id']]
        num1 = pd.DataFrame(data=content_ratings[0:, 0:])
        frames = [num, num1]

        cb = pd.concat(frames, axis=1)
        cb.columns = ['book_id', 'content_rating']

        print(cb.columns)
        print("# of content ratings computed:", len(cb['content_rating']))
        return cb

    ##### final_hybrid ######
    def final_hybrid(self, md):
        print("#################### final_hybrid ####################")
        hyb = md[['book_id']]
        title = md[['book_id', 'title']]
        hyb = hyb.merge(title, on='book_id')
        hyb = hyb.merge(self.ratings, on='book_id')
        hyb = hyb.merge(self.collaborative_rating, on='book_id')
        #hyb = hyb.merge(self.popularity_rating, on='book_id')
        hyb = hyb.merge(self.content_rating, on='book_id')
        hyb = hyb[['book_id', 'title', 'rating', 'cf_rating', 'content_rating']]

        def weighted_rating(x):
            cf = x['cf_rating'] * 0.4
            #pr = x['popularity_rating'] * 0.2
            cb = x['content_rating'] * 0.4
            return cf + cb

        hyb['hyb_rating'] = hyb.apply(weighted_rating, axis=1)
        hyb = hyb.sort_values('hyb_rating', ascending=False).head(999)

        # after all the work is done, drop rows of item already rated book ids by user_id
        user_alreadyRatedBookId = ratings[(ratings['user_id'] == self.user_id)][['book_id']]
        # Get all indexes
        indexNames = hyb[hyb['book_id'].isin(user_alreadyRatedBookId['book_id'].tolist())].index
        #print("user_alreadyRatedBookId: ", user_alreadyRatedBookId['book_id'].tolist())
        #print("hyb_book_ids: ", hyb['book_id'].tolist())
        # Delete these row indexes from dataFrame
        hyb.drop(indexNames, inplace=True)
        hyb.drop_duplicates('book_id', inplace=True)

        hyb.columns = ['Book ID', 'Title', 'Original', 'Collaborative', 'Content', 'Hybrid']
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print("# of hybrid ratings computed:", len(hyb['Hybrid']))
        print("\nShow ratings grid: \n", hyb)

ratings = pd.read_csv('../data/original/CustomData/ratings.csv')
ratings = ratings[1:10000]
# taking only the first 100000 ratings
userId = 25
print('\n----------------Welcome User ' + str(userId) + '-------------------')
h = hybrid(userId, ratings)