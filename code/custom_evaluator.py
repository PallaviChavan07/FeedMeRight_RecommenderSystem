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

    def get_recipes_interacted(self, user_id):
        # Get the user's data and merge in the information.
        try:
            interacted_items = self.interactions_full_indexed_df.loc[user_id]
        except:
            interacted_items = None
        return interacted_items

    def evaluate_model_for_user_enchanced(self, model, user_id):
        users_recs_df = model.recommend_items(user_id, items_to_ignore=[], topn=10000000000)
        #print(user_id, " Size of users_recs_df: ", users_recs_df.shape)

        if users_recs_df is None:
            return {'recall@5': 0, 'interacted_count': 0, 'precision@5': 0, 'accuracy@5': 0}

        ks = [5, 10, 20] #list of all ks we want to try
        recall = {} #create dictionaries
        precision = {}
        accuracy = {}
        n_rel = {}
        for k in ks:
            # get top k recos for the user from the complete users_cb_recs_df
            user_top_k_recos = users_recs_df.head(k)

            # get recipes already interacted by user
            user_interact_recipes_df = self.get_recipes_interacted(user_id)
            # print("user_interact_recipes_df: ", len(user_interact_recipes_df), " for user_id ", user_id)

            # filter out recipes with rating > 3.5 which is our threshold for good vs bad recipes
            user_interated_relevant_df = user_interact_recipes_df.loc[user_interact_recipes_df['rating'] >= 3.0]
            user_interated_irrelevant_df = user_interact_recipes_df.loc[user_interact_recipes_df['rating'] < 3.0]
            # print("user_interated_relevant_df: ", len(user_interated_relevant_df))

            # merge top k recommended recipes with filtered user interacted recipes to get relevant recommended
            relevant_and_reco_items_df = user_top_k_recos.merge(user_interated_relevant_df, how='inner', on='recipe_id')
            # print("relevant_and_reco_items_df:\n", relevant_and_reco_items_df)

            irrelevant_and_reco_items_df = user_top_k_recos.merge(user_interated_irrelevant_df, how='inner',
                                                                  on='recipe_id')
            # user_top_k_recos_count = len(user_top_k_recos)
            # p_recall = len(relevant_and_reco_items_df) / user_top_k_recos_count if user_top_k_recos_count != 0 else 1
            # print("Pallavi dumb recall", p_recall)

            # Recall@K: Proportion of relevant items that are recommended
            n_rel_and_rec_k = len(relevant_and_reco_items_df)  # TP
            n_rel[k] = len(user_interated_relevant_df)
            n_irrel_and_rec_k = len(irrelevant_and_reco_items_df)  # TN
            recall[k] = n_rel_and_rec_k / n_rel[k] if n_rel[k] != 0 else 1
            # print("amod yet to correct but dumb recall", a_recall)

            # Number of recommended items in top k (Whose score is higher than 0.5 (relevant))
            n_rec_k = len(user_top_k_recos.loc[user_top_k_recos['recStrength'] >= 0.3])
            # Precision@K: Proportion of recommended items that are relevant
            precision[k] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            accuracy[k] = (n_rel_and_rec_k + n_irrel_and_rec_k) / k


        person_metrics = {'recall@5': recall[5], 'interacted_count': n_rel[5], 'precision@5': precision[5], 'accuracy@5': accuracy[5],
                          'recall@10': recall[10], 'interacted_count': n_rel[10], 'precision@10': precision[10], 'accuracy@10': accuracy[10],
                          'recall@20': recall[20], 'interacted_count': n_rel[20], 'precision@20': precision[20], 'accuracy@20': accuracy[20]}

        # print(person_metrics)
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        #if model.get_model_name() == 'ContentBased':
        users_metrics = []
        for idx, user_id in enumerate(list(self.interactions_full_indexed_df.index.unique().values)):
            singleuser_metric = self.evaluate_model_for_user_enchanced(model, user_id)
            users_metrics.append(singleuser_metric)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(users_metrics).sort_values('interacted_count', ascending=False)
        global_recall_5 = detailed_results_df['recall@5'].sum() / len(detailed_results_df['recall@5'])
        global_precision_5 = detailed_results_df['precision@5'].sum() / len(detailed_results_df['precision@5'])
        global_accuracy_5 = detailed_results_df['accuracy@5'].sum() / len(detailed_results_df['accuracy@5'])

        global_recall_10 = detailed_results_df['recall@10'].sum() / len(detailed_results_df['recall@10'])
        global_precision_10 = detailed_results_df['precision@10'].sum() / len(detailed_results_df['precision@10'])
        global_accuracy_10 = detailed_results_df['accuracy@10'].sum() / len(detailed_results_df['accuracy@10'])

        global_recall_20 = detailed_results_df['recall@20'].sum() / len(detailed_results_df['recall@20'])
        global_precision_20 = detailed_results_df['precision@20'].sum() / len(detailed_results_df['precision@20'])
        global_accuracy_20 = detailed_results_df['accuracy@20'].sum() / len(detailed_results_df['accuracy@20'])

        global_metrics = {'model': model.get_model_name(), 'recall@5': global_recall_5, 'precision@5': global_precision_5, 'accuracy@5': global_accuracy_5,
                          'recall@10': global_recall_10, 'precision@10': global_precision_10, 'accuracy@10': global_accuracy_10,
                          'recall@20': global_recall_20, 'precision@20': global_precision_20, 'accuracy@20': global_accuracy_20}
        detailed_results_df = detailed_results_df
        # else:
        #     people_metrics = []
        #     for idx, user_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
        #         # if idx % 100 == 0 and idx > 0:
        #         #    print('%d users processed' % idx)
        #         person_metrics = self.evaluate_model_for_user(model, user_id)
        #         person_metrics['_user_id'] = user_id
        #         people_metrics.append(person_metrics)
        #     print('%d users processed' % idx)
        #
        #     detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        #
        #     global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
        #         detailed_results_df['interacted_count'].sum())
        #     global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
        #         detailed_results_df['interacted_count'].sum())
        #
        #     global_metrics = {'modelName': model.get_model_name(), 'recall@5': global_recall_at_5,
        #                       'recall@10': global_recall_at_10}


        return global_metrics, detailed_results_df
