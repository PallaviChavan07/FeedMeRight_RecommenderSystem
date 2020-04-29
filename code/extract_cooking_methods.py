# Source - https://github.com/yangli53/happy_meals/blob/master/LDA.ipynb

import pandas as pd
import numpy as np
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
nltk.download(['stopwords','wordnet'])
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LdaMulticore, CoherenceModel
import tqdm
import pyLDAvis
import pyLDAvis.gensim
from pprint import pprint
df = pd.read_csv('../data/small_10k/export_rated_recipes_set.csv')
#df.head(1)

cooking_methods_corpus = ['al dente','bake','barbecue','baste','batter','beat','blanch','blend','boil','broil','caramelize','chop','clarify','cream','cure','deglaze','degrease','dice','dissolve','dredge','drizzle','dust','fillet','flake','flambe','fold','fricassee','fry','garnish','glaze','grate','gratin','grill','grind','julienne','knead','lukewarm','marinate','meuniere','mince','mix','pan-broil','pan-fry','parboil','pare','peel','pickle','pinch','pit','planked','plump','poach','puree','reduce','refresh','render','roast','saute','scald','scallop','score','sear','shred','sift','simmer','skim','steam','steep','sterilize','stew','stir','toss','truss','whip']

print("df shape = ", df.shape)
# check num of recipes
#df.shape

# keep title and ingredients for text cleaning
df = df[['recipe_id','recipe_name', 'cooking_directions']]
#df.cooking_directions[0]
print("df shape = ", df.shape)
# create a list for text data
print("len of df['cooking_directions'] = ", len(df['cooking_directions']))
directions_dict_lst = df['cooking_directions'].tolist()
print("directions_dict_lst len = ", len(directions_dict_lst))

directions_lst = []
for direction_dict in directions_dict_lst:
    direction_dict = ast.literal_eval(direction_dict)
    #print("direction['directions'] = ", direction_dict['directions'])
    directions_lst.append(direction_dict['directions'])

data = directions_lst[:5]
#print("Actual data = ", data)
cooking_methods_list = []
for data in directions_lst:
    #function to split text into word
    tokens = word_tokenize(data)
    nltk.download('stopwords')
    #print("After tokenization", tokens)

    # remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    #print("remove punctuation with nltk ",words)

    # filter out stop words

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #print(words)

    # stemming of words

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    #print("stemmed = ",stemmed)
    clean_list = [each_string.lower() for each_string in stemmed]
    common = list(set(clean_list) & set(cooking_methods_corpus))
    recipe_cooking_methods = ""
    #print("For new recipe")
    for i in common:
        #print (i)
        recipe_cooking_methods = recipe_cooking_methods+" "+i
    #print(recipe_cooking_methods.strip())
    cooking_methods_list.append(recipe_cooking_methods.strip())

print("size of df = = ", len(df))
print("size of cooking_methods_list list = ", len(cooking_methods_list))
cooking_methods_list = ["nan" if x == '' else x for x in cooking_methods_list]

while("" in cooking_methods_list) :
    cooking_methods_list.remove("")
print("Empty cooking_methods_list list = ", '' in cooking_methods_list)
print("size of cooking_methods_list list = ", len(cooking_methods_list))


# remove number and punctuation
# no_punctuation_lst = [re.sub(r'[^a-zA-Z]', ' ', sent.lower()) for sent in directions_lst]
# print("data[0] = ", directions_lst[0])

# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield(gensim.utils.simple_preprocess(sentence))
#
# data_words = list(sent_to_words(directions_list))
# print("data_words[0] = ",data_words[0])
#
# # build a bigram model
# bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
# # faster way to get a sentence clubbed as a bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# data_bigram = [bigram_mod[doc] for doc in data_words]
# print("data_bigram[0] = ",data_bigram[0])
# stop_words = stopwords.words('english')
#
# # remove stopwords
# data_nonstop = [[word for word in doc if word not in stop_words] for doc in data_bigram]
# print("data_nonstop[0] = ", data_nonstop[0])
#
# def lemmatization(texts):
#     wordnet_lemmatizer = WordNetLemmatizer()
#     texts_out = []
#     print("texts = ", texts)
#     for w in texts:
#         print("Sentence in texts = ", w)
#         lemmatized_output = ' '.join([wordnet_lemmatizer.lemmatize(w)])
#         print("lemmatized output for text = ", lemmatized_output)
#         #print("{0:20}{1:20}".format(word, wordnet_lemmatizer.lemmatize(word, pos="v")))
#         #for token in sent,wordnet_lemmatizer.lemmatize(token, pos="v"):
#         texts_out.append(lemmatized_output)
#     return texts_out
# lemmatized = lemmatization(data_nonstop[0])
# print("lemmatized = ", lemmatized)

# self-define stopwords
# recipe_stopwords = ['slice','large','diced','stock','taste','leave','powder','sliced','fresh','low','fat',
#                     'whole','purpose','mix','ripe','medium','raw','coarse','style','active','dry','ground',
#                     'white','heart','piece','crushed','cut','minute','pocket','shredded','optional','cube',
#                     'hour','bag','baby','seeded','small','clove','country','leaf','dressing','center','fillet',
#                     'sea','chunk','light','food','head','container','link','frozen','can','cooked','stalk',
#                     'regular','dusting','heavy','round','rinsed','thawed','jar','solid','creamy','z','fluid',
#                     'uncooked','sheet','strip','short','soft','mixed','blue','flake','warm','unbleached','sun',
#                     'old','topping','wedge','thick','lean','extra','meal','preserve','mild','half','crosswise',
#                     'new','seasoning','kidney','black','green','red','yellow','white','unpeeled','boiling',
#                     'amount','cold','snow','cluster','necessary','firm','soda','cubed','temperature','deep',
#                     'flat','iron','seedless','boneless','strong','bottle','unsweetended','smoked','melted',
#                     'thin','hard','pure','bulk','unsalted','deveined','petite','cooking','box','prepared',
#                     'softened','split','kosher','blanched_slivered','carton','canned','flavor','broken','free',
#                     'blend','lengthwise','real','purple','dice','flaked','bite_sized','refrigerated_crescent',
#                     'reserved','undrained','original','stuffing','bulb','sharp','reduced_fat','color',
#                     'pressed','diagonal','good','season','bit','jumbo','instant','skim','chopped','paper',
#                     'towel','roasted','flaky','ear','flavoring','fine','minced', 'square','size','single',
#                     'refrigerated','skinless','pitted','bay','seasoned','divided','long','crumbled','filling',
#                     'miniature','mashed','peeled','top','bottom','flat_leaf','rubbed','liquid','ready','chop',
#                     'non','frying','condensed','stewed','light','food','container','link','can','optional',
#                     'diced','fluid','meal','preserve','seasoning','bottle','box','split','flavor','lengthwise',
#                     'flavoring','square','size','at_room','grade','shape','cuisine']
#
# # remove self-defined stopwords
# data_clean = [[word for word in doc if word not in recipe_stopwords] for doc in data_nonstop]
# print("data_clean[0] = ", data_clean)
# clean_ingredients = []
# for list in data_clean:
#     clean_ingredients.append(' '.join(list))
# print("clean_ingredients = ", clean_ingredients)
#
# df['cleaned_ingredients'] = clean_ingredients
#
# print(df.head(2))
# df.to_csv(r'../data/original/recipes_cleaningredients.csv', index = False)


