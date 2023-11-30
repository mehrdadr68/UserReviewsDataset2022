import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore') # setting ignore as a parameter
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from distutils.dir_util import copy_tree
from time import daylight
from test import get_predictions
from locale import normalize
import numpy as np 
import spacy
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet
from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import AffinityPropagation
from IPython.display import display
import os
import unicodedata
from copy import deepcopy
import sys
import copy
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from nltk.stem import WordNetLemmatizer
from string import digits
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
import nltk
from nltk.stem import WordNetLemmatizer 
from itertools import product
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from operator import itemgetter
from statistics import median
from transformers import pipeline
from time import sleep
from tqdm import tqdm
from itertools import groupby
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
with open('negative-words.txt') as f:
    list_negated = f.read().splitlines()
def sentiment_text_analysis(text):
    model = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")
    predicted = model(text)
    var = predicted[0]
    temp = list(var.values())
    return temp[0]
def Text_Summerize_Bart(text):
    summerize = summarizer(text,min_length=10,max_length=len(text.split()),do_sample=False)
    var_str = str(summerize[0])
    var_str = var_str.replace("summary_text': ", '')
    var_str = var_str.replace("{", '')
    var_str = var_str.replace("}", '')
    var_str = var_str.replace("'", '')
    return var_str
list_featurrequest = []
list_bugfix = []
def data_cleaning(Data):
        print("Cleaning Data : \n")
        idx = len(Data.values)
        stop_words = set(stopwords.words('english'))
        stopwords_doc = []
        englishwords = []
        tokens = []
        tempdocss = []
        transacts = []
        print("Read Data Frame Values : ")
        for i in tqdm(range(0, idx)):
            temp2 = (Data.values[i][0])
            temp2 = str(temp2)
            strtemp1 = temp2.replace("'","")
            strtemp2 = strtemp1.replace("]","")
            strtemp3 = strtemp2.replace("[","")
            transacts.append(strtemp3)
        print("Tokenizing User Review Sentences : ")
        for item in tqdm(range(0,len(transacts))):
            tokens.append(word_tokenize(transacts[item].lower()))
        Stem = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        print("Lematizing User Review Sentences : ")
        for i in tqdm(range(0,len(tokens))):
            tokentemp = []
            for j in range(0,len(tokens[i])):
                tokentemp.append(str(lemmatizer.lemmatize(tokens[i][j])))
            tempdocss.append(copy.deepcopy(tokentemp))
        print("Removing Stop Words in User Reviews : ")
        for i in tqdm(range(0,len(tempdocss))):
            filtered_text = []
            for j in range(0,len(tempdocss[i])):
                token = tempdocss[i][j]
                if token not in stop_words:
                    filtered_text.append(copy.deepcopy(token))
            stopwords_doc.append(copy.deepcopy(filtered_text)) 
        print("Removing Non-English Words in User Reviews : ")
        for i in tqdm(range(0,len(stopwords_doc))): # Remove Non English Tokens
            filtered_text = []
            for j in range(0,len(stopwords_doc[i])):
                token = stopwords_doc[i][j]
                if not wordnet.synsets(token):
                    pass
                else:
                    filtered_text.append(copy.deepcopy(token))
            englishwords.append(copy.deepcopy(filtered_text)) 
        nonumberTokens= []
        print("Removing Numbers in User Reviews : ")
        for i in tqdm(range(0,len(englishwords))): # # Remove Numbers in Comments
            filtered_text = []
            for j in range(0,len(englishwords[i])):
                tempstring = englishwords[i][j]
                remove_digits = str.maketrans('', '', digits)
                res = tempstring.translate(remove_digits)
                filtered_text.append(copy.deepcopy(res))
            nonumberTokens.append(copy.deepcopy(filtered_text))
        to_export = []
        print("Saving Results : ")
        for i in tqdm(range (0,len(nonumberTokens))): 
            temp = " ".join(str(x) for x in nonumberTokens[i])
            to_export.append(copy.deepcopy(temp))
        return to_export
def data_cleaning_without_lematize(Data):
        idx = len(Data.values)
        stop_words = set(stopwords.words('english'))
        stopwords_doc = []
        englishwords = []
        tokens = []
        tempdocss = []
        transacts = []
        for i in range(0, idx):
            temp2 = (Data.values[i][0])
            temp2 = str(temp2)
            strtemp1 = temp2.replace("'","")
            strtemp2 = strtemp1.replace("]","")
            strtemp3 = strtemp2.replace("[","")
            transacts.append(strtemp3)
        for item in range(0,len(transacts)):
            tokens.append(word_tokenize(transacts[item].lower()))
        for i in range(0,len(tokens)): # Remove Non English Tokens
            filtered_text = []
            for j in range(0,len(tokens[i])):
                token = tokens[i][j]
                if not wordnet.synsets(token):
                    pass
                else:
                    filtered_text.append(copy.deepcopy(token))
            englishwords.append(copy.deepcopy(filtered_text)) 
        to_export = []
        for i in range (0,len(englishwords)): 
            temp = " ".join(str(x) for x in englishwords[i])
            to_export.append(copy.deepcopy(temp))
        return to_export
def GroupByListOfWords(text,keywordlist): #check if list of word in sentence (for clustring based on words)
    text = text.lower()
    word_list = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    keys = set(keywordlist)
    var = [" ".join(grp) for res, grp in groupby(lemmatized_output.split(), keys.__contains__) if res]
    return var
def detect_negated_words(text):
    text_list = text.split(" ") #convert text to list for index
    flag_has_verb = 0
    has_verb = False
    for i in range (0,len(list_negated)):
        word = str(list_negated[i])
        if word in text_list:
            for i in range (0,len(text_list)):
                if (word == text_list[i]):
                    negated_index = i
            doc = nlp(text)
            for token in doc:
                has_verb = token.pos_ == "VERB"
                if has_verb == True:
                    verb = token
                    flag_has_verb = 1
                    break
            if(flag_has_verb == 0):
                no_verb_sen =[]
                no_verb_sen.append(text_list[negated_index])
                no_verb_sen.append(text_list[negated_index + 1])
                break

    if (has_verb == True):
        return  str(verb)
    else:
        return ''
Data_thumbsup = pd.read_csv('reviews.csv',usecols =["content","thumbsUpCount"])
comment_cleaned = data_cleaning(Data_thumbsup)
print("Classifying User Reviews : ")
for i in tqdm(range (0,len(comment_cleaned))):
    templist = []
    string = comment_cleaned[i]
    prediction = get_predictions(string)
    if ((prediction) == 'featurerequest'):
        templist = []
        templist.append(copy.deepcopy(string))
        templist.append(copy.deepcopy(Data_thumbsup.values[i][1])) #append thumbs up of comment
        templist.append(copy.deepcopy(i))
        templist.append(copy.deepcopy(len(string)))
        list_featurrequest.append(copy.deepcopy(templist))
    elif ((prediction) == 'bugfix'):
        templist = []
        templist.append(copy.deepcopy(string))
        templist.append(copy.deepcopy(Data_thumbsup.values[i][1])) #append thumbs up of comment
        templist.append(copy.deepcopy(i))
        templist.append(copy.deepcopy(len(string)))
        list_bugfix.append(copy.deepcopy(templist))
sorted_list_featurerequst = sorted(list_featurrequest, key=itemgetter(1),reverse=True)
sorted_list_bugfix = sorted(list_bugfix, key=itemgetter(1),reverse=True)
thumbsup_feature = []
thumbup_bugfix = []
print("Appending ThumbsUp of User Reviews to List : ")
for i in tqdm(range(0,len(sorted_list_featurerequst))):
    if(sorted_list_featurerequst[i][1] != 0):
        thumbsup_feature.append(sorted_list_featurerequst[i][1])
for i in tqdm(range(0,len(sorted_list_bugfix))):
    if(sorted_list_bugfix[i][1] !=0):
        thumbup_bugfix.append(sorted_list_bugfix[i][1])
median_feature = median(thumbsup_feature)
median_bugfix = median(thumbup_bugfix)
priority_list_feature = []
priority_list_feature_summerized = []
priority_list_bugfix = []
priority_list_bugfix_summerized = []
for i in range(0,len(sorted_list_featurerequst)):
    if(sorted_list_featurerequst[i][1] > median_feature):
        priority_list_feature.append(copy.deepcopy(sorted_list_featurerequst[i]))
for i in range(0,len(sorted_list_bugfix)):
    if(sorted_list_bugfix[i][1] > median_bugfix):
        priority_list_bugfix.append(copy.deepcopy(sorted_list_bugfix[i]))
print("Reading Reviews Again : ")
rawdata = pd.read_csv('reviews.csv',usecols =["content"])
review_without_lemiatize = []
print("Removing [,],' from Reviews : ")
idx = len(rawdata.values)
for i in tqdm(range(0, idx)):
    temp2 = (rawdata.values[i][0])
    temp2 = str(temp2)
    strtemp1 = temp2.replace("'","")
    strtemp2 = strtemp1.replace("]","")
    strtemp3 = strtemp2.replace("[","")
    review_without_lemiatize.append(strtemp3)
with open('./Fun-NonFunc/Compatibility_issue.txt') as f:
    list_compatiblity_keyword = f.read().splitlines()
with open('./Fun-NonFunc/Networkproblem_connection.txt') as f:
    list_network_keyword = f.read().splitlines()
with open('./Fun-NonFunc/Privacy_security.txt') as f:
    list_privacy_security_keyword = f.read().splitlines()
with open('./Fun-NonFunc/Resource_Consumption.txt') as f:
    list_resource_consumption_keyword = f.read().splitlines()
with open('./Fun-NonFunc/Update_issue.txt') as f:
    list_update_issue_keyword = f.read().splitlines()
with open('./Fun-NonFunc/usability.txt') as f:
    list_usability_keyword = f.read().splitlines()
print("Summrizing Feature Request Type Reviews : ")
list_compatiblity = []
list_network = []
list_privacy_security = []
list_resource_consumption = []
list_update_issue = []
list_usability = []
for i in tqdm(range(0,len(priority_list_feature))):
    index = priority_list_feature[i][2]
    text = review_without_lemiatize[index]
    temp_list = []
    temp_list.append(copy.deepcopy(text))
    temp_list.append(copy.deepcopy(priority_list_feature[i][1]))
    summerized_text = Text_Summerize_Bart(text)
    var = GroupByListOfWords(text,list_compatiblity_keyword)
    if (len(var) != 0):
        list_compatiblity.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_network_keyword)
    if (len(var) != 0):
        list_network.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_privacy_security_keyword)
    if (len(var) != 0):
        list_privacy_security.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_resource_consumption_keyword)
    if (len(var) != 0):
        list_resource_consumption.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_update_issue_keyword)
    if (len(var) != 0):
        list_update_issue.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_usability_keyword)
    if (len(var) != 0):
        list_usability.append(copy.deepcopy(summerized_text))
    temp_list.append(copy.deepcopy(summerized_text))
    sen = sentiment_text_analysis(summerized_text)
    temp_list.append(copy.deepcopy(sen))
    priority_list_feature_summerized.append(copy.deepcopy(temp_list))
print("Summrizing Bug Fix Type Reviews : ")
for i in tqdm(range(0,len(priority_list_bugfix))):
    index = priority_list_bugfix[i][2]
    text = review_without_lemiatize[index]
    temp_list = []
    temp_list.append(copy.deepcopy(text))
    temp_list.append(copy.deepcopy(priority_list_bugfix[i][1]))
    summerized_text = Text_Summerize_Bart(text)
    var = GroupByListOfWords(text,list_compatiblity_keyword)
    if (len(var) != 0):
        list_compatiblity.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_network_keyword)
    if (len(var) != 0):
        list_network.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_privacy_security_keyword)
    if (len(var) != 0):
        list_privacy_security.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_resource_consumption_keyword)
    if (len(var) != 0):
        list_resource_consumption.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_update_issue_keyword)
    if (len(var) != 0):
        list_update_issue.append(copy.deepcopy(summerized_text))
    var = GroupByListOfWords(text,list_usability_keyword)
    if (len(var) != 0):
        list_usability.append(copy.deepcopy(summerized_text))
    temp_list.append(copy.deepcopy(summerized_text))
    sen = sentiment_text_analysis(summerized_text)
    temp_list.append(copy.deepcopy(sen))
    priority_list_bugfix_summerized.append(copy.deepcopy(temp_list))
df_feature = pd.DataFrame(priority_list_feature_summerized)
df_feature.to_csv('featurerequest.csv')
df_bug = pd.DataFrame(priority_list_bugfix_summerized)
df_bug.to_csv('bugfix.csv')
#,columns=['Number', 'Review','ThumbsUp','Review_Summrized','Sentiment/Polarity']
df = pd.DataFrame(list_compatiblity)
df.to_csv('compatiblity.csv')
df = pd.DataFrame(list_network)
df.to_csv('network.csv')
df = pd.DataFrame(list_privacy_security)
df.to_csv('privacy_security.csv')
df = pd.DataFrame(list_resource_consumption)
df.to_csv('resource_consumption.csv')
df = pd.DataFrame(list_update_issue)
df.to_csv('update_issue.csv')
df = pd.DataFrame(list_usability)
df.to_csv('usability.csv')
print("Hello")