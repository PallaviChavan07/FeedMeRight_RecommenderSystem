import pandas as pd
import random

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:
    def __init__(self, recipe_df, interactions_full_indexed_df=None, interactions_train_indexed_df=None, interactions_test_indexed_df=None):
        self.recipe_df = recipe_df
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df

    @staticmethod
    def get_items_interacted(user_id, passeddf):
        # Get the user's data and merge in the information.
        try:
            interacted_items = passeddf.loc[user_id]['recipe_id']
        except:
            interacted_items = None

        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = self.get_items_interacted(user_id, self.interactions_full_indexed_df)
        all_items = set(self.recipe_df['recipe_id'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    @staticmethod
    def _verify_hit_top_n(item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, user_id):
        # Getting the items in test set
        interacted_values_testset = self.interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_testset['recipe_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['recipe_id'])
        else:
            person_interacted_items_testset = {int(interacted_values_testset['recipe_id'])}
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(user_id, items_to_ignore=self.get_items_interacted(user_id, self.interactions_train_indexed_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items, create a set
            items_to_filter_recs = non_interacted_items_sample.union({item_id})

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            if not person_recs_df is None:
                valid_recs_df = person_recs_df[person_recs_df['recipe_id'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['recipe_id'].values
            else:
                #this way we can still get person_metrics
                valid_recs = None

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}

        #if person_recs_df is None: print(person_metrics)
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        if model.get_model_name() == 'ContentBased':
            users_metrics = []
            for idx, user_id in enumerate(list(self.interactions_full_indexed_df.index.unique().values)):
                users_recs_df = model.recommend_items(user_id, items_to_ignore=[], topn=10000000000)
                # print(user_id, " Size of users_recs_df: ", users_recs_df.shape)
                singleuser_metric = model.cb_evaluate_model_for_user(user_id, users_recs_df, k=5)
                users_metrics.append(singleuser_metric)
            print('%d users processed' % idx)
            print('\nusers_metrics: ', len(users_metrics))

            detailed_results_df = pd.DataFrame(users_metrics).sort_values('interacted_count', ascending=False)
            global_recall = detailed_results_df['recall@5'].sum() / len(detailed_results_df['recall@5'])
            global_precision = detailed_results_df['precision@5'].sum() / len(detailed_results_df['precision@5'])
            global_accuracy = detailed_results_df['accuracy@5'].sum() / len(detailed_results_df['accuracy@5'])

            global_metrics = {'modelName': model.get_model_name(), 'recall@5': global_recall, 'precision@5': global_precision, 'accuracy@5': global_accuracy}
            detailed_results_df = detailed_results_df
        else:
            people_metrics = []
            for idx, user_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
                # if idx % 100 == 0 and idx > 0:
                #    print('%d users processed' % idx)
                person_metrics = self.evaluate_model_for_user(model, user_id)
                person_metrics['_user_id'] = user_id
                people_metrics.append(person_metrics)
            print('%d users processed' % idx)

            detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

            global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
                detailed_results_df['interacted_count'].sum())
            global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
                detailed_results_df['interacted_count'].sum())

            global_metrics = {'modelName': model.get_model_name(), 'recall@5': global_recall_at_5,
                              'recall@10': global_recall_at_10}


        return global_metrics, detailed_results_df
