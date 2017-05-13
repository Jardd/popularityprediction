# -*- coding: utf-8 -*-
import numpy as np
import string
from nltk import PorterStemmer
import sys  
import analyse
from sklearn import svm

reload(sys)  
sys.setdefaultencoding('utf8')
""" 
value <= threshild = 0
value > threshold =1
"""
""" headlines: ei dom objekt mit allen objekten"""


####Training####
def make_no_dub_list(seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    #print result
    return result
   
def make_array(noDub, headlines):
    all_vektors=[]
    for headline in headlines:
        headline_vektor=[0] * len(noDub)
        
        text=headline.getElementsByTagName("text")[0].childNodes[0].data.encode('utf-8')
        text=text.split()
        for w in text:
            
            for char in string.punctuation:
                w = w.replace(char, '')
            
            w=PorterStemmer().stem_word(w)
            w = w.strip().lower()
            
            if w in noDub:
                headline_vektor[noDub.index(w)]=1
        all_vektors.append(headline_vektor)
    return all_vektors
                
    



def train_arrays(headlines, tag):
    threshold=analyse.find_median(analyse.make_list_num(headlines, tag))
    full_text=[]
    classes=[]
    for headline in headlines:
        text=headline.getElementsByTagName("text")[0].childNodes[0].data.encode('utf-8')
        value=int(headline.getElementsByTagName(tag)[0].childNodes[0].data)
        if value<=threshold:
            classes.append(0)
        else:
            classes.append(1)
        
    #get rif of all punctuation        
        for char in string.punctuation:
            text = text.replace(char, '')
        
        for w in text.split():
            w=PorterStemmer().stem_word(w)
            full_text.append(w.strip().lower())

    noDub=make_no_dub_list(full_text)
    #noDub=make_no_dub_list(full_text, lambda x: x.lower())
    all_vektors=make_array(noDub, headlines)
    #print len(all_vektors), "  ,   " , len(classes)
    return all_vektors, classes, noDub



    
def fit_training_svm(all_vektors, all_classes):
    clf = svm.SVC()
    #all_vektors=np.array(all_vektors)
    #all_classes=np.array(all_classes)
    clf.fit(all_vektors, all_classes) 
    return clf
    
    """ Classifing"""
def test_arrays(headlines, noDub, tag):
    all_vektors=make_array(noDub, headlines)
    threshold=analyse.find_median(analyse.make_list_num(headlines, tag))
    gold_classes=[]
    for headline in headlines:
        value=int(headline.getElementsByTagName(tag)[0].childNodes[0].data)
        if value<=threshold:
            gold_classes.append(0)
        else:
            gold_classes.append(1)
    return all_vektors, gold_classes
    




