import string
import re

import json
import nltk
from xml.dom.minidom import parse
import xml.dom.minidom
import unicodedata

def remove_punctuation(text):
    punctutation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
    return ''.join(x for x in text
                   if unicodedata.category(x) not in punctutation_cats)
def stem(headlines_list_text):
    #stopwords=nltk.corpus.stopwords.words('english')
    new_headlines_list_text=[]
    for headline in headlines_list_text:
        new_headline=[]
        #print headline
        for word in headline:
     #       if not word in stopwords:
            word=remove_punctuation(word)
            word=nltk.PorterStemmer().stem_word(word)
            new_headline.append(word)
        new_headlines_list_text.append(new_headline)
    return new_headlines_list_text

def lowerCase(headlines_list_text):
    new_headlines_list_text=[]
    for headline in headlines_list_text:
        new_headline=[]
        for word in headline:
            word=word.strip().lower()
            new_headline.append(word)
        new_headlines_list_text.append(new_headline)
    return new_headlines_list_text

def getWordsFromJson(headlines_json):
    new_headlines=[]
    for headline in headlines_json:
        new_headline=[]
        for tup in headline:
            new_headline.append(tup[0].lower())
        new_headlines.append(new_headline)
    return new_headlines

def posTagging(headlines):
    new_headlines=[]
    for headline in headlines:
        text=headline.getElementsByTagName("text")[0].childNodes[0].data
        text = nltk.word_tokenize(text)
        text_pos=nltk.pos_tag(text)
        print text_pos
        new_headlines.append(text_pos)
    return new_headlines

def lemmatizer(text_pos):
    new_headlines=[]
    lemmatizer = nltk.WordNetLemmatizer()
    for headline in text_pos:
        new_headline=[]
        for word_pos in headline:
            pos=get_wordnet_pos(word_pos[1])
            if pos != "":
                new_headline.append(lemmatizer.lemmatize(word_pos[0].lower(), pos))
            else:
                new_headline.append(word_pos[0].lower())
        new_headlines.append(new_headline)
    return new_headlines

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''
        
def posToJson(headlines_train,train_name, headlines_test, test_name):
    train_headlines_list_text=posTagging(headlines_train)
    with open(train_name+'.json', 'wb') as outfile:
        json.dump(train_headlines_list_text, outfile)
    
    test_headlines_list_text=posTagging(headlines_test)
    with open(test_name+'.json', 'wb') as outfile:
        json.dump(test_headlines_list_text, outfile)

def removeStopwords(features):
    stopwords=nltk.corpus.stopwords.words('english')
    new_features=[]
    for f in features:
        if f not in stopwords:
            new_features.add(f)
    return new_features
