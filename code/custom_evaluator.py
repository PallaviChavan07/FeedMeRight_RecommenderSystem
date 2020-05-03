import pandas as pd

class ModelEvaluator:
    def __init__(self, recipe_df, interactions_full_indexed_df=None, interactions_train_indexed_df=None, interactions_test_indexed_df=None):
        self.recipe_df = recipe_df
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df

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

        if users_recs_df is None: return {'recall@5': 0, 'interacted_count': 0, 'precision@5': 0, 'accuracy@5': 0}

        ks = [5, 10, 20] #list of all ks we want to try
        recall = {} #create dictionaries
        precision = {}
        accuracy = {}
        f1score = {}
        n_rel = {}
        for k in ks:
            # get top k recos for the user from the complete users_cb_recs_df
            user_top_k_recos = users_recs_df.head(k)

            # get only items with recStrength > 0.5 i.e threshold
            user_top_k_recos = user_top_k_recos.loc[user_top_k_recos['recStrength'] >= 0.5]

            # get recipes already interacted by user
            user_interact_recipes_df = self.get_recipes_interacted(user_id)
            # print("user_interact_recipes_df: ", len(user_interact_recipes_df), " for user_id ", user_id)

            # filter out recipes with rating > 3.5 which is our threshold for good vs bad recipes
            user_interated_relevant_df = user_interact_recipes_df.loc[user_interact_recipes_df['rating'] >= 3]
            user_interated_irrelevant_df = user_interact_recipes_df.loc[user_interact_recipes_df['rating'] < 3]
            # print("user_interated_relevant_df: ", len(user_interated_relevant_df))

            # merge top k recommended recipes with filtered user interacted recipes to get relevant recommended
            relevant_and_reco_items_df = user_top_k_recos.merge(user_interated_relevant_df, how='inner', on='recipe_id')
            # print("relevant_and_reco_items_df:\n", relevant_and_reco_items_df)

            irrelevant_and_reco_items_df = user_top_k_recos.merge(user_interated_irrelevant_df, how='inner',on='recipe_id')
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
            n_rec_k = len(user_top_k_recos.loc[user_top_k_recos['recStrength'] >= 0.5])
            # Precision@K: Proportion of recommended items that are relevant
            precision[k] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            accuracy[k] = (n_rel_and_rec_k + n_irrel_and_rec_k) / k

            f1score[k] = 2 * ((precision[k] * recall[k]) / (precision[k] + recall[k])) if (precision[k] + recall[k] > 0.0) else 0


        person_metrics = {'recall@5': recall[5], 'precision@5': precision[5], 'accuracy@5': accuracy[5], 'f1score@5': f1score[5],
                          'recall@10': recall[10], 'precision@10': precision[10], 'accuracy@10': accuracy[10], 'f1score@10': f1score[10],
                          'recall@20': recall[20], 'precision@20': precision[20], 'accuracy@20': accuracy[20], 'f1score@20': f1score[20]}

        # print(person_metrics)
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        #if model.get_model_name() == 'ContentBased':
        users_metrics = []
        #for idx, user_id in enumerate(list(self.interactions_full_indexed_df.index.unique().values)):
        for idx, user_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
            singleuser_metric = self.evaluate_model_for_user_enchanced(model, user_id)
            users_metrics.append(singleuser_metric)
        #print('%d users processed' % idx)

        #detailed_results_df = pd.DataFrame(users_metrics).sort_values('interacted_count', ascending=False)
        detailed_results_df = pd.DataFrame(users_metrics)
        global_recall_5 = detailed_results_df['recall@5'].sum() / len(detailed_results_df['recall@5'])
        global_precision_5 = detailed_results_df['precision@5'].sum() / len(detailed_results_df['precision@5'])
        global_accuracy_5 = detailed_results_df['accuracy@5'].sum() / len(detailed_results_df['accuracy@5'])
        global_f1score_5 = detailed_results_df['f1score@5'].sum() / len(detailed_results_df['f1score@5'])

        global_recall_10 = detailed_results_df['recall@10'].sum() / len(detailed_results_df['recall@10'])
        global_precision_10 = detailed_results_df['precision@10'].sum() / len(detailed_results_df['precision@10'])
        global_accuracy_10 = detailed_results_df['accuracy@10'].sum() / len(detailed_results_df['accuracy@10'])
        global_f1score_10 = detailed_results_df['f1score@10'].sum() / len(detailed_results_df['f1score@10'])

        global_recall_20 = detailed_results_df['recall@20'].sum() / len(detailed_results_df['recall@20'])
        global_precision_20 = detailed_results_df['precision@20'].sum() / len(detailed_results_df['precision@20'])
        global_accuracy_20 = detailed_results_df['accuracy@20'].sum() / len(detailed_results_df['accuracy@20'])
        global_f1score_20 = detailed_results_df['f1score@20'].sum() / len(detailed_results_df['f1score@20'])

        global_metrics = {'model': model.get_model_name(), 'recall@5': global_recall_5, 'precision@5': global_precision_5, 'accuracy@5': global_accuracy_5, 'f1score@5': global_f1score_5,
                          'recall@10': global_recall_10, 'precision@10': global_precision_10, 'accuracy@10': global_accuracy_10, 'f1score@10': global_f1score_10,
                          'recall@20': global_recall_20, 'precision@20': global_precision_20, 'accuracy@20': global_accuracy_20, 'f1score@20': global_f1score_20}

        return global_metrics, detailed_results_df
