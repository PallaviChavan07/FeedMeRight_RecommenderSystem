### TODO
### Check how many ratings are given by each user.. and select maybe only those users who have given approximately similar number of recipes.

#Extract train test recipes from toalt set of recipes. Because we want to work on only recipes which are rated by user.
import nltk
import pandas as pd
import json
import ast
import numpy as np
import os
from random import seed
from random import randint
nltk.download(['stopwords','wordnet'])
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
stop_words = stopwords.words('english')
import spacy

pd.set_option("display.max_rows", None, "display.max_columns", None)
train_ratings_df = pd.read_csv(os.path.realpath('../data/original/core-data-train_rating.csv'))
test_ratings_df = pd.read_csv(os.path.realpath('../data/original/core-data-test_rating.csv'))
train_test_ratings_df = pd.concat([train_ratings_df, test_ratings_df], ignore_index=True)
core_recipes_df = pd.read_csv(os.path.realpath('../data/original/core-data_recipe.csv'))

def get_rated_recipes(core_recipes_df):
    # Get all recipe ids from all interactions
    interaction_recipe_ids = train_test_ratings_df.recipe_id.unique()
    # Get all unique recipes from core recipes set
    all_unique_recipe_ids = core_recipes_df.recipe_id.unique()
    # Common recipes from interacted recipes and actual recipe data = all rated recipes
    rated_recipe_ids = list(set(interaction_recipe_ids) & set(all_unique_recipe_ids))
    rated_recipes_df = core_recipes_df.loc[core_recipes_df['recipe_id'].isin(rated_recipe_ids)]
    valid_interactions_df = train_test_ratings_df.loc[train_test_ratings_df['recipe_id'].isin(rated_recipe_ids)]

    # print("Actual number of recipes = ", len(core_recipes_df))
    # print("all unique recipes = ", len(all_unique_recipe_ids))
    # print("rated recipes = ", len(rated_recipes_df))
    # print("interacted unique recipes = ", len(interaction_recipe_ids))
    # print("Number of interactions = ", len(train_test_ratings_df))
    # print("valid interacted recipes = ", len(valid_interactions_df))
    return rated_recipes_df, valid_interactions_df

# def get_unique_users():
#
#     print("user-recipe-rating interactions = ", len(ratings_df))
#     dup_users_df = pd.DataFrame()
#     users_df = pd.DataFrame()
#     dup_users_df['user_id'] = ratings_df['user_id'].to_numpy()
#     print("Number of users = ", len(dup_users_df))
#     users_df['user_id'] = ratings_df['user_id'].unique()
#     print("Number of unique users = ", len(users_df ))
#     return users_df


def user_data_generation(rated_recie_df, ratings_df):
    sedentary = 1
    lightly_active = 2
    moderately_active = 3
    very_active = 4
    extra_active = 5
    sedentary_mf = 1.2
    lightly_active_mf = 1.375
    moderately_active_mf = 1.55
    very_active_mf = 1.725
    extra_active_mf = 1.9

    # Minimum count of users rated recipes
    # Eka recipe la kamit kami 50 users ne rating dila aahe. # 1 recipe to 50 recipes
    count_of_users_rated_atleas_50 = 50
    filter_recipes = ratings_df['recipe_id'].value_counts() > count_of_users_rated_atleas_50

    print("filter_recipes = ", len(filter_recipes))
    filter_recipes = filter_recipes[filter_recipes].index.tolist()
    #
    MIN_USER_INTERACTION = 10
    MAX_USER_INTERACTION = 25
    filter_users = (ratings_df['user_id'].value_counts() > MIN_USER_INTERACTION) & (
                ratings_df['user_id'].value_counts() <= MAX_USER_INTERACTION)
    print("filter_users ", len(filter_users))
    filter_users = filter_users[filter_users].index.tolist()
    filtered_ratings_df = ratings_df[
        (ratings_df['recipe_id'].isin(filter_recipes)) & (ratings_df['user_id'].isin(filter_users))]
    print("length of filtered_ratings_df = ", len(filtered_ratings_df))
    print("length of unique users in filtered_ratings_df = ", len(filtered_ratings_df.user_id.unique()))
    #######

    # users_interactions_count_df = ratings_df.groupby(['user_id', 'recipe_id']).size().groupby('user_id').size()
    # MIN_USERS_INTERACTIONS = 20
    # # MAX_USERS_INTERACTIONS = 120
    # # users_with_enough_interactions_df = users_interactions_count_df[ (users_interactions_count_df >= MIN_USERS_INTERACTIONS) &
    # #                                     (users_interactions_count_df < MAX_USERS_INTERACTIONS)].reset_index()[['user_id']]
    # users_with_enough_interactions_df = users_interactions_count_df[(users_interactions_count_df >= MIN_USERS_INTERACTIONS)].reset_index()[['user_id']]
    user_df = pd.DataFrame()
    #user_df['user_id'] = list(users_with_enough_interactions_df['user_id'].head(10000))
    user_df['user_id'] = list(filtered_ratings_df.user_id.unique())[:10000]
    print("user_df = ", user_df.shape)
    print(user_df.head())
    weight_height_df = pd.read_csv(os.path.realpath('../data/original/weight-height.csv'))
    print(user_df.shape[0])
    print(weight_height_df.shape[0])
    user_df['Gender'] = weight_height_df['Gender'].to_list()

    #print("length of usrs - ", len(users_df), "\n htwt = ", len(weight_height_df))
    user_df['Height_inch'] = list(weight_height_df['Height_inch'])
    user_df['Weight_lb'] = list(weight_height_df['Weight_lb'])
    # convert height to meter
    height_mtr =  weight_height_df['Height_inch']*0.0254
    user_df['height_mtr'] = weight_height_df['height_mtr'] = list(height_mtr)
    # conver weight to kgs
    weight_kgs = weight_height_df['Weight_lb']*0.453592
    user_df['weight_kgs'] = weight_height_df['weight_kgs'] = list(weight_kgs)
    #print(weight_height_df.columns.values)
    bmi = weight_height_df['weight_kgs'] / np.power((weight_height_df['height_mtr']),2)
    user_df['BMI'] = list(bmi)

    #print("bmi",bmi)
    # generate random integer values

    # seed random number generator
    seed(1)
    # generate some integers
    age_list = []
    activity_list = []
    for _ in range(len(user_df)):
        age_list.append(randint(18, 40))
        activity_list.append(randint(1, 5))
    user_df['age'] = age_list
    user_df['activity'] =activity_list

    user_df['BMR'] = user_df.apply(
        lambda row: 66 + (6.3 * row.Weight_lb) + (12.9 * row.Height_inch) - (6.8 * row.age) if (row.Gender == 'Male')
        else 655 + (4.3 * row.Weight_lb) + (4.7 * row.Height_inch) - (4.7 * row.age), axis = 1)

    user_df['calories_per_day'] = user_df.apply(
        lambda row: row.BMR * sedentary_mf  if (row.activity == sedentary)
        else row.BMR * lightly_active_mf if (row.activity == lightly_active)
        else row.BMR * moderately_active_mf  if (row.activity == moderately_active)
        else row.BMR * very_active_mf if (row.activity == very_active)
        else row.BMR * extra_active_mf, axis=1)

    #user_df.to_csv(r'../data/original/clean/users.csv', index=False, header=True)
    # print("height_mtr = ", height_mtr)
    # print("weight_kgs = ", weight_kgs)
    return user_df

def get_clean_ingredients(core_recipe_df):
    ingredients_lst = core_recipe_df['ingredients'].tolist()
    # create a list for text data
    # data = df.ingredients.values.tolist()

    recipe_stopwords = ['slice', 'large', 'diced', 'stock', 'taste', 'leave', 'powder', 'sliced', 'fresh', 'low', 'fat',
                        'whole', 'purpose', 'mix', 'ripe', 'medium', 'raw', 'coarse', 'style', 'active', 'dry',
                        'ground',
                        'white', 'heart', 'piece', 'crushed', 'cut', 'minute', 'pocket', 'shredded', 'optional', 'cube',
                        'hour', 'bag', 'baby', 'seeded', 'small', 'clove', 'country', 'leaf', 'dressing', 'center',
                        'fillet',
                        'sea', 'chunk', 'light', 'food', 'head', 'container', 'link', 'frozen', 'can', 'cooked',
                        'stalk',
                        'regular', 'dusting', 'heavy', 'round', 'rinsed', 'thawed', 'jar', 'solid', 'creamy', 'z',
                        'fluid',
                        'uncooked', 'sheet', 'strip', 'short', 'soft', 'mixed', 'blue', 'flake', 'warm', 'unbleached',
                        'sun',
                        'old', 'topping', 'wedge', 'thick', 'lean', 'extra', 'meal', 'preserve', 'mild', 'half',
                        'crosswise',
                        'new', 'seasoning', 'kidney', 'black', 'green', 'red', 'yellow', 'white', 'unpeeled', 'boiling',
                        'amount', 'cold', 'snow', 'cluster', 'necessary', 'firm', 'soda', 'cubed', 'temperature',
                        'deep',
                        'flat', 'iron', 'seedless', 'boneless', 'strong', 'bottle', 'unsweetended', 'smoked', 'melted',
                        'thin', 'hard', 'pure', 'bulk', 'unsalted', 'deveined', 'petite', 'cooking', 'box', 'prepared',
                        'softened', 'split', 'kosher', 'blanched_slivered', 'carton', 'canned', 'flavor', 'broken',
                        'free',
                        'blend', 'lengthwise', 'real', 'purple', 'dice', 'flaked', 'bite_sized',
                        'refrigerated_crescent',
                        'reserved', 'undrained', 'original', 'stuffing', 'bulb', 'sharp', 'reduced_fat', 'color',
                        'pressed', 'diagonal', 'good', 'season', 'bit', 'jumbo', 'instant', 'skim', 'chopped', 'paper',
                        'towel', 'roasted', 'flaky', 'ear', 'flavoring', 'fine', 'minced', 'square', 'size', 'single',
                        'refrigerated', 'skinless', 'pitted', 'bay', 'seasoned', 'divided', 'long', 'crumbled',
                        'filling',
                        'miniature', 'mashed', 'peeled', 'top', 'bottom', 'flat_leaf', 'rubbed', 'liquid', 'ready',
                        'chop',
                        'non', 'frying', 'condensed', 'stewed', 'light', 'food', 'container', 'link', 'can', 'optional',
                        'diced', 'fluid', 'meal', 'preserve', 'seasoning', 'bottle', 'box', 'split', 'flavor',
                        'lengthwise',
                        'flavoring', 'square', 'size', 'at_room', 'grade', 'shape', 'cuisine', 'hot', 'water', 'salt']
    # remove units
    units = ['spoon', 'pound', 'cup', 'quart', 'ounce', 'pint', 'degree', 'dash', 'pinch', 'pack', 'halve', 'bunch',
             'inch']

    # keep only noun and adj
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
        texts_out = []
        for sent in texts:
            doc = nlp(' '.join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            # word_lst = []
            # for word in sent : word_lst.append(lemma.lemmatize(word))
            # texts_out.append(word_lst)
        return texts_out

    # initialize spacy 'en' model, keeping only tagger component
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    tokenized_ingredients_lst = []
    for data in ingredients_lst:
        # remove number and punctuation

        # print("data before splitting = ", data)
        data = data.split("^")
        data = [re.sub(r'[^a-zA-Z]', ' ', sent.lower()) for sent in data]
        for unit in units:
            expre = re.compile('\w*' + unit + '\w*')
            data = [re.sub(expre, '', sent).strip() for sent in data]
        # remove duplicated whitespace
        data = [' '.join(sent.split()) for sent in data]
        # print("data after splitting = ", data)
        ingredstr = ""
        for ingred in data:
            ingredstr += " " + ingred
        ingredstr = ingredstr.strip().lower()
        # print("ingredstr = ",ingredstr)
        # function to split text into word
        tokens = nltk.word_tokenize(ingredstr)
        # nltk.download('stopwords')
        # print("After tokenization", tokens)
        tokenized_ingredients_lst.append(tokens)
        # print("tokenized_ingredients_lst = ", tokenized_ingredients_lst)

        # porter = nltk.PorterStemmer()
        # stemmed = [porter.stem(word) for word in tokens]
    data_nonstop = [[word for word in doc if word not in stop_words] for doc in tokenized_ingredients_lst]
    #print("\n data_nonstop = ", data_nonstop)
    # initialize spacy 'en' model, keeping only tagger component
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    # lemmatize , allowed_postags=['NOUN', 'ADJ']
    data_lemmatized = lemmatization(data_nonstop)
    #print("\n data_lemmatized = ", data_lemmatized)
    data_clean = [[word for word in doc if word not in recipe_stopwords] for doc in data_lemmatized]
    #print("\n data_clean[0] = ", data_clean)

    clean_ingredients_list = []
    for ingred_lst in data_clean:
        ingredstr = ""
        for ingred in ingred_lst:
            ingredstr += " " + ingred
        clean_ingredients_list.append(ingredstr.strip())
    #print("clean_ingredients_list \n", clean_ingredients_list)
    #core_recipe_df.drop(['ingredients'], inplace=True,axis=1)
    core_recipe_df['clean_ingredients'] = clean_ingredients_list
    return core_recipe_df


def drop_recipes_with_no_calories():
    core_recipes_df = pd.read_csv(os.path.realpath('../data/original/core-data_recipe.csv'))
    nutritions_lst = core_recipes_df['nutritions'].tolist()
    calories_list = []
    for nut in nutritions_lst:
        nut = ast.literal_eval(nut)
        calories_list.append(nut['calories']['amount'])
    core_recipes_df['calories'] = calories_list
    print("bfore removing 0 calories ", len(core_recipes_df))
    core_recipes_df = core_recipes_df[core_recipes_df['calories'] != 0]
    print("After removing 0 calories ", len(core_recipes_df))
    #print("drop_recipes_with_no_calories returned columns ######## ", )
    return core_recipes_df
    #recipe_df.to_csv(r'../data/original/recipes_with_calories.csv', index=False, header=True)

def get_recipes_with_cook_methods(core_recipe_df):
    cooking_methods_corpus = ['al dente', 'bake', 'barbecue', 'baste', 'batter', 'beat', 'blanch', 'blend', 'boil',
                              'broil', 'caramelize', 'chop', 'clarify', 'cream', 'cure', 'deglaze', 'degrease', 'dice',
                              'dissolve', 'dredge', 'drizzle', 'dust', 'fillet', 'flake', 'flambe', 'fold', 'fricassee',
                              'fry', 'garnish', 'glaze', 'grate', 'gratin', 'grill', 'grind', 'julienne', 'knead',
                              'lukewarm', 'marinate', 'meuniere', 'mince', 'mix', 'pan-broil', 'pan-fry', 'parboil',
                              'pare', 'peel', 'pickle', 'pinch', 'pit', 'planked', 'plump', 'poach', 'puree', 'reduce',
                              'refresh', 'render', 'roast', 'saute', 'scald', 'scallop', 'score', 'sear', 'shred',
                              'sift', 'simmer', 'skim', 'steam', 'steep', 'sterilize', 'stew', 'stir', 'toss', 'truss',
                              'whip']
    directions_dict_lst = core_recipe_df['cooking_directions'].tolist()
    print("directions_dict_lst len = ", len(directions_dict_lst))

    directions_lst = []
    for direction_dict in directions_dict_lst:
        direction_dict = ast.literal_eval(direction_dict)
        directions_lst.append(direction_dict['directions'])
    cooking_methods_list = []
    for data in directions_lst:
        # function to split text into word
        tokens = nltk.word_tokenize(data)
        #nltk.download('stopwords')
        # print("After tokenization", tokens)
        # remove all tokens that are not alphabetic
        words = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # print(words)
        # stemming of words
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        # print("stemmed = ",stemmed)
        clean_list = [each_string.lower() for each_string in stemmed]
        common = list(set(clean_list) & set(cooking_methods_corpus))
        recipe_cooking_methods = ""
        for i in common:
            recipe_cooking_methods = recipe_cooking_methods + " " + i
        cooking_methods_list.append(recipe_cooking_methods.strip())
    cooking_methods_list = ["nothing" if x == '' else x for x in cooking_methods_list]
    core_recipe_df.drop(['cooking_directions'], inplace=True, axis=1)
    core_recipe_df['cook_method'] = cooking_methods_list
    return core_recipe_df

def get_health_labels(recipes_with_cook_methods):
    nutritions_lst = recipes_with_cook_methods['nutritions'].tolist()
    diet_lables_list = []
    for nut in nutritions_lst:
        diet_labels = ""
        nut = ast.literal_eval(nut)
        try:
            if int(nut['protein']['percentDailyValue']) > 20: diet_labels += ("highprotein")
        except:
            None
        try:
            if int(nut['fiber']['percentDailyValue']) > 20: diet_labels += (" highfiber")
        except:
            None
        try:
            if int(nut['fat']['percentDailyValue']) < 5: diet_labels += (" lowfat")
        except:
            None
        try:
            if int(nut['carbohydrates']['percentDailyValue']) < 5: diet_labels += (" lowcarb")
        except:
            None
        try:
            if int(nut['sodium']['percentDailyValue']) < 5: diet_labels += (" lowsodium")
        except:
            None
        if diet_labels is "": diet_labels = "balanced"
        diet_lables_list.append(diet_labels)

    #print("diet_lables_list = ", diet_lables_list)
    #print("df size = ", len(recipes_with_cook_methods))
    #print("diet_lables_list sie = ", len(diet_lables_list))
    recipes_with_cook_methods.drop(['nutritions'], inplace=True, axis=1)

    recipes_with_cook_methods['diet_labels'] = diet_lables_list
    return recipes_with_cook_methods

if __name__ == '__main__':
    core_recipes_df = drop_recipes_with_no_calories()
    recipes_with_clean_ingredients_df = get_clean_ingredients(core_recipes_df)
    recipes_with_cook_methods = get_recipes_with_cook_methods(recipes_with_clean_ingredients_df)
    recipes_with_health_labels = get_health_labels(recipes_with_cook_methods)
    rated_recie_df, valid_interactions_df = get_rated_recipes(recipes_with_health_labels)
    user_df = user_data_generation(rated_recie_df, valid_interactions_df)
    rated_recie_df.to_csv(os.path.realpath(r'../data/clean/recipes.csv'), index=False, header=True)
    valid_interactions_df.to_csv(os.path.realpath(r'../data/clean/ratings.csv'), index=False, header=True)
    user_df.to_csv(os.path.realpath(r'../data/clean/users.csv'), index=False, header=True)

