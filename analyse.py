# -*- coding: utf-8 -*-

#import sys
import corpora
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from collections import Counter
def find_max(list_of_likes):
    return max(list_of_likes)

def find_min(list_of_likes):
    return min(list_of_likes)
    
def find_mean(list_of_likes):
    return sum(list_of_likes)/float(len(list_of_likes))
    
def find_median(list_of_likes):
    list_of_likes.sort()
    return np.median(list_of_likes)
    
    
def make_list_num(headlines, tag):
    liste=[]
    for headline in headlines:
       # print headline.getElementsByTagName(tag)[0]
        item=headline.getElementsByTagName(tag)[0]
        liste.append(int(item.childNodes[0].data.encode('utf-8')))
    return liste
    
def print_stats(headlines, tag):
    print "***",tag,"***"
    liste=make_list_num(headlines, tag)
    print tag+" max: ",find_max(liste)
    print tag+" min: ",find_min(liste)
    print tag+" avg: ",find_mean(liste)
    print tag+" Median: ", find_median(liste)
    
def boxplot(data_liste):
        plt.figure()
        plt.boxplot(data_liste , 0, 'rs', 0)
        plt.show()
def writeInFile(data_liste):
        occ=Counter(data_liste)
        f=open("Ergebnisse\\verteilung_guard_train.csv", "w")
        f.write("tweets\tocc\n")
        for key, value in occ.items():
            line=str(key)+"\t"+str(value)+"\n"
            f.write(line)
        f.close

def clfResult(train_corpus, test_corpus, tag, headlines_train, fs, test_classes, pred_classes, target_names):
    print "Train Set: " +train_corpus
    print "Test Set: "+test_corpus
    print "Tag: " + tag
    print_stats(headlines_train, tag)
    print "Feature Selection:" , fs
    print "++++++++++++++++++++++++++++++++++++++++++\n"
    print "\n"
    print sklearn.metrics.classification_report(test_classes, pred_classes, target_names=target_names)
    print sklearn.metrics.confusion_matrix(test_classes, pred_classes)
    print "Accuracy: ",sklearn.metrics.accuracy_score(test_classes, pred_classes)
    
def getDist2(headlines, classes):
    print "Dokumente insgesammt: ", len(headlines)
    #print nClass , " Klassen"
    a=0
    b=0
    c=0
    d=0
    for i in classes:
        if i ==0:
           a+=1
        elif i==1:
            b+=1
        elif i==2:
            c+=1
        else:
            d+=1
    print "0: ", a
    print "1: ", b
    print "2: ", c
    print "3: ", d
