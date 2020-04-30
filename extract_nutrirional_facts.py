#“balanced”, “high-protein”, “high-fiber”, “low-fat”, “low-carb”, “low-sodium”
import nltk
import pandas as pd
import json
import ast
import numpy as np
from random import seed
from random import randint
nltk.download(['stopwords','wordnet'])
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
core_recipes_df = pd.read_csv('../data/original/core-data_recipe.csv')
nutritions_lst = core_recipes_df['nutritions'].tolist()
diet_lables_list = []
for nut in nutritions_lst:
    diet_labels = ""
    nut = ast.literal_eval(nut)
    try:
        if int(nut['protein']['percentDailyValue']) > 20 : diet_labels+=("highprotein")
    except: None
    try:
        if int(nut['fiber']['percentDailyValue']) > 20 : diet_labels+=(" highfiber")
    except:None
    try:
        if int(nut['fat']['percentDailyValue']) < 5 : diet_labels+=(" lowfat")
    except:None
    try:
        if int(nut['carbohydrates']['percentDailyValue']) < 5 : diet_labels+=(" lowcarb")
    except:None
    try:
        if int(nut['sodium']['percentDailyValue']) < 5 : diet_labels+=(" lowsodium")
    except:None
    if diet_labels is "": diet_labels = "balanced"
    diet_lables_list.append(diet_labels)

print("diet_lables_list = ",diet_lables_list)
print("df size = ", len(core_recipes_df))
print("list sie = ", len(diet_lables_list))
# core_recipes_df['diet_labels'] = diet_lables_list
# print("bfore removing 0 calories ", len(core_recipes_df))
# core_recipes_df = core_recipes_df[core_recipes_df['calories'] != 0]
# print("After removing 0 calories ", len(core_recipes_df))
# return core_recipes_df