
import analyse
import math
import sys
import nltk
import textProcessing
import vektorMaker
import featureSelection
import json
import xmlProc
from sklearn.feature_extraction.text  import CountVectorizer
import ml
import os
from collections import defaultdict 
import numpy as np





def unigramBaseline(train_corpus, test_corpus,split, tag, fs, k):
    """Runs the unigram Baseline.  
    train_corpus: path to the train corpus in XML format
    test_corpus: path to the test corpus in XML format
    split: number of Klasses (either 2 or 4)
    tag: XML tag of popularity messure (T1 for tweets after one day)
    fs: Type of Featureselection. ("CHI" for chi-squarred Error or "" empty sting if no feature selection )
    k: numbers of features to select.
    """
    headlines_train , headlines_test =xmlProc.readXML(train_corpus, test_corpus)
    
    
    #print len(headlines_train) , " : ",len (headlines_test)
    #print headlines_train[0].getElementsByTagName("text")[0].childNodes[0].data
    #print headlines_test[0].getElementsByTagName("text")[0].childNodes[0].data
    #headlines_train=textProcessing.posToJson(headlines_train,train_corpus+".json",headlines_test, test_corpus+".json")

    
    #nur bei aenderung des testsets aufrufen
    #new_train_corpus=corpora.remove_and_quotes(r"corpora\nyt_test.xml", "changed_nyt_test.xml")
    #new_test_corpus=corpora.remove_and_quotes(r"corpora\nyt_training.xml", "changed_nyt_train.xml")
    #textProcessing.posToJson(headlines_train,train_corpus, headlines_test, test_corpus)
    
    with open(train_corpus+'.json') as data_file:    
        train_headlines_list_text= json.load(data_file)
    with open(test_corpus+'.json') as data_file:
        test_headlines_list_text= json.load(data_file)
        
    if split ==2:
        train_classes, threshold=vektorMaker.get2ClassesTrain(headlines_train, tag)
        test_classes=vektorMaker.get2ClassesTest(headlines_test, tag,threshold)
    elif split ==4 or split==44:
        train_classes, threshold=vektorMaker.get4ClassesTrain(headlines_train, tag)
        test_classes=vektorMaker.get4ClassesTest(headlines_test, tag, threshold)
    else:
        print "Fehler invalide classen anzahl"
        
    
    
    train_headlines_list_text=textProcessing.getWordsFromJson(train_headlines_list_text)
    test_headlines_list_text=textProcessing.getWordsFromJson(test_headlines_list_text)
    print "Train Stats:"
    analyse.getDist2(train_headlines_list_text, train_classes)
    print train_headlines_list_text[0]
    #print len(train_headlines_list_text)
    print "\n"
    print "Test Stats:"
    analyse.getDist2(test_headlines_list_text, test_classes)
    print test_headlines_list_text[0]
    
    
    #low vs hig
    if split ==44:
        high_low_classes=[]
        high_low_vektors=[]
        for i in range(0, len(train_classes)):
            if train_classes[i]==0 or train_classes[i]==3:
                high_low_classes.append(train_classes[i])
                high_low_vektors.append(train_headlines_list_text[i])
        train_headlines_list_text=high_low_vektors
        train_classes=high_low_classes
        
        high_low_classes=[]
        high_low_vektors=[]
        for i in range(0, len(test_classes)):
            if test_classes[i]==0 or test_classes[i]==1:
                high_low_classes.append(0)
                high_low_vektors.append(test_headlines_list_text[i])
            if test_classes[i]==2 or test_classes[i]==3:
                high_low_classes.append(3)
                high_low_vektors.append(test_headlines_list_text[i])
        test_headlines_list_text=high_low_vektors
        test_classes=high_low_classes
    #print len(train_headlines_list_text)
    #print train_headlines_list_text
    
    
    
    
    
    
    """Feature Selection"""
    #features=featureSelection.all_features(train_headlines_list_text)
    #print len(features)
    #features=textProcessing.removeStopwords(features)
   # print "feature selection.."
    #features=featureSelection.PMI2(train_classes, train_headlines_list_text, 0)
    
    """Create Vektor"""
    
    train_headlines_list_text=textProcessing.stem(train_headlines_list_text)
    test_headlines_list_text=textProcessing.stem(test_headlines_list_text)
    
    train_headlines_str_text=vektorMaker.listToString(train_headlines_list_text)
    test_headlines_str_text=vektorMaker.listToString(test_headlines_list_text)
    vectorizer = CountVectorizer()
    train_headlines_vektor_binary=vectorizer.fit_transform(train_headlines_str_text)
    test_headlines_vektor_binary=vectorizer.transform(test_headlines_str_text)
    
    features = vectorizer.get_feature_names()

    if "SVC" in fs:
        #print "featureSelection SVC..."
        train_headlines_vektor_binary, test_headlines_vektor_binary = featureSelection.SVCfeature_seletion(train_headlines_vektor_binary, test_headlines_vektor_binary, train_classes)
    if "CHI" in fs:
        #print "featureSelection CHI..."
        train_headlines_vektor_binary, test_headlines_vektor_binary=featureSelection.chiSquare(train_headlines_vektor_binary, test_headlines_vektor_binary, train_classes, k, features)
        
    if "MI" in fs:
        #print "featureSelection MI..."
        train_headlines_vektor_binary, test_headlines_vektor_binary=featureSelection.MI(train_headlines_vektor_binary, test_headlines_vektor_binary, train_classes, k)
    
    target_names=["low","med-low","med-high","high"]
    #print "Training.."
    clf = ml.fit_training_svm(train_headlines_vektor_binary, train_classes)
    #print "predicting.."
    pred_classes=clf.predict(test_headlines_vektor_binary)
    #for i in range(0, len(test_headlines_list_text)):
    #    print test_classes[i], "\t", pred_classes[i] ,"\t",test_headlines_list_text[i] , "\t", "contorl"

    analyse.clfResult(train_corpus, test_corpus, tag, headlines_train, fs, test_classes, pred_classes, target_names)





def skipgram(train_corpus, test_corpus,split, tag, embeddings_json, konkatenation_method,ext_feature_file_train, ext_feature_file_test):
    """Uses pretrained Embeddings to classify headlines
      
    train_corpus: path to the train corpus in XML format
    test_corpus: path to the test corpus in XML format
    split: number of Klasses (either 2 or 4)
    tag: XML tag of popularity messure (T1 for tweets after one day)
    embeddings_json: Pre-trained Embeddings in dict format in a json file or a text file with "word \t embedding".
    ext_feature_file_train: path to file which holds the external features from Dimitrova et al. for the trainingsset
    ext_feature_file_test: path to file which holds the external features from Dimitrova et al. for the testset
    """
    headlines_train , headlines_test =xmlProc.readXML(train_corpus, test_corpus)
    corpus_train=train_corpus.split("/")[1]  
    corpus_test=test_corpus.split("/")[1]  
    print embeddings_json   
    with open(train_corpus+'.json') as data_file:    
        train_headlines_list = json.load(data_file)
    with open(test_corpus+'.json') as data_file:
        test_headlines_list = json.load(data_file)
    if embeddings_json.endswith(".json"):
        print "in if"
        with open(embeddings_json) as data_file:
            embeddings_dict=json.load(data_file)
    else:
        embeddings_dict=loadGloVe(embeddings_json)
      
    if split ==2:
        train_classes, threshold=vektorMaker.get2ClassesTrain(headlines_train, tag)
        test_classes=vektorMaker.get2ClassesTest(headlines_test, tag,threshold)
    elif split ==4 or split==44:
        train_classes, threshold=vektorMaker.get4ClassesTrain(headlines_train, tag)
        test_classes=vektorMaker.get4ClassesTest(headlines_test, tag, threshold)
    else:
        print "Fehler invalide classen anzahl"
        
    
    
    train_headlines_list_text=textProcessing.getWordsFromJson(train_headlines_list)
    test_headlines_list_text=textProcessing.getWordsFromJson(test_headlines_list)

    
    if split ==4:
        high_low_classes=[]
        high_low_vektors=[]
        for i in range(0, len(train_classes)):
            if train_classes[i]==0 or train_classes[i]==3:
                high_low_classes.append(train_classes[i])
                high_low_vektors.append(train_headlines_list_text[i])
        train_headlines_list_text=high_low_vektors
        train_classes=high_low_classes
        
        high_low_classes=[]
        high_low_vektors=[]
        for i in range(0, len(test_classes)):
            if test_classes[i]==0 or test_classes[i]==1:
                high_low_classes.append(0)
                high_low_vektors.append(test_headlines_list_text[i])
            if test_classes[i]==2 or test_classes[i]==3:
                high_low_classes.append(3)
                high_low_vektors.append(test_headlines_list_text[i])
        test_headlines_list_text=high_low_vektors
        test_classes=high_low_classes
    print len(train_headlines_list_text)
    print len(train_classes)

    check=len(embeddings_dict["for"])    

    
    ###Make words verktor to embedding vektor####
    global oov
    oov={}
    print "start making train_emb"
    i=0
    embedd=embeddings_json.split("/")[1]
    headlines_emb_train="headlines_"+konkatenation_method+"_"+corpus_train+"_"+embedd
    headlines_emb_test="headlines_"+konkatenation_method+"_"+corpus_test+"_"+embedd
    if os.path.exists("headline_new/"+headlines_emb_train): #headlines already representet as embeddings with weights applied
        with open("headline_new/"+headlines_emb_train) as data_file:
            train_matrix=json.load(data_file)
    else:
        if konkatenation_method == "add":
            train_matrix=convert_headlines_to_emb_add(train_headlines_list_text, embeddings_dict)
        elif konkatenation_method == "multi":
            train_matrix=headline_emb_multi(train_headlines_list_text, embeddings_dict)
        elif konkatenation_method == "addEXT":
            train_matrix=convert_headlines_to_emb_EXT(train_headlines_list_text, embeddings_dict,ext_feature_file_train)
        elif konkatenation_method == "multiEXT" :
            train_matrix=headline_emb_multi_EXT(train_headlines_list_text, embeddings_dict,ext_feature_file_train)
        else:
            print "invalide konkatenations methode"
        with open("headline_new/"+headlines_emb_train, "w") as f:
            json.dump(train_matrix, f)
                          

    print "train to emb finished"

    print len(train_matrix)
    test_matrix=[]
    print "stat making test_emb"
    
    if os.path.exists("headline_new/"+headlines_emb_test): #headlines already representet as embeddings with weights applied
        with open("headline_new/"+headlines_emb_test) as data_file:
            test_matrix=json.load(data_file)
    else:
        if konkatenation_method == "add" :
            test_matrix=convert_headlines_to_emb_add(test_headlines_list_text, embeddings_dict)
            
        elif konkatenation_method == "multi" :
            test_matrix=headline_emb_multi(test_headlines_list_text, embeddings_dict)
        
        elif konkatenation_method == "addEXT":
            test_matrix=convert_headlines_to_emb_EXT(test_headlines_list_text, embeddings_dict,ext_feature_file_test)
        
        elif konkatenation_method == "multiEXT" :
            test_matrix=headline_emb_multi_EXT(test_headlines_list_text, embeddings_dict,ext_feature_file_test)
        else:
            print "invalide konkatenations methode"
            
            
        with open("headline_new/"+headlines_emb_test, "w") as f:
            json.dump(test_matrix, f)
    print "test to emb finished"
   

    print len(test_headlines_list_text)
    print len(test_classes)
    print len(test_matrix)
    target_names=["low","high"]
    #train_matrix, test_matrix=featureSelection.chiSquare(train_matrix, test_matrix, train_classes, 170, [])
    if split==44:
        target_names=["low","low-med","high-med","high"]
    print "Training.."
    #print train_matrix.shape
    clf = ml.fit_training_svm(train_matrix, train_classes)
    print "predicting.."
    pred_classes=clf.predict(test_matrix)
    
    fs=[]
    analyse.clfResult(train_corpus, test_corpus, tag, headlines_train, fs, test_classes, pred_classes, target_names)
    print "Mehode:", konkatenation_method
    print "EMB:", embeddings_json



def create_idf_dict(headlines_list_text):
    """
    Given the headlines of the trainingsset, creates an idf dictionary "{word: idfscore}"
    headlines_list_text: a 2D list of tokenized headlines 
    returns: dict
    """
    
    
    vocab=set(sum(headlines_list_text, []))
    idf=defaultdict(int)
    num_doc=len(headlines_list_text)
   # number of doc ,number of doc with term
    num_doc=len(headlines_list_text)
    for word in vocab:
        for headline in headlines_list_text:
            if word in headline:
                idf[word]=idf[word]+1
    for key in idf:
        idf[key]=math.log((num_doc/float(idf[key])),2)
    
    return idf



def convert_headlines_to_emb_EXT(headlines_list_text, embeddings_dict, ext_features_file):
    """
    
    
    """
    ext_features=read_ext_features(ext_features_file)
    
    matrix=[]
    check=len(embeddings_dict["for"])    
    i=0
    for headline in headlines_list_text:
        
        sum_vektor_h=np.zeros(check)
        for word in headline:
            
            if word in embeddings_dict.keys():
                sum_vektor_h=sum_vektor_h+(np.array(embeddings_dict[word]))
            elif word in oov:
                sum_vektor_h=sum_vektor_h+(oov[word])
            else:
                oov[word]=np.random.rand(len(sum_vektor_h))
                sum_vektor_h=sum_vektor_h+(oov[word])

        #print len(sum_vektor_h)
        if len(sum_vektor_h)==check:
            print >> sys.stderr, i , ' von ', len(headlines_list_text)
            matrix.append(sum_vektor_h.tolist()+ext_features[i])                
        else:
            print "sollte nicht mehr passieren!!"
    #        print word
            matrix.append(np.zeros(check).tolist())
        i=i+1
    return matrix

def convert_headlines_to_emb_add(headlines_list_text, embeddings_dict):
    matrix=[]
    check=len(embeddings_dict["for"])    
    i=0
    for headline in headlines_list_text:
        
        sum_vektor_h=np.zeros(check)
        for word in headline:
            
            if word in embeddings_dict.keys():
                sum_vektor_h=sum_vektor_h+(np.array(embeddings_dict[word]))
            elif word in oov:
                sum_vektor_h=sum_vektor_h+(oov[word])
            else:
                oov[word]=np.random.rand(len(sum_vektor_h))
                sum_vektor_h=sum_vektor_h+(oov[word])

        #print len(sum_vektor_h)
        if len(sum_vektor_h)==check:
            print >> sys.stderr, i , ' von ', len(headlines_list_text)
            matrix.append(sum_vektor_h.tolist())                
        else:
            print "sollte nicht mehr passieren!!"
    #        print word
            matrix.append(np.zeros(check).tolist())
        i=i+1
    return matrix

def loadGloVe(filename):
    vocab = []
    embd = []
    emb_dict={}
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        
        emb=map(float, row[1:])
        emb_dict[row[0].decode("utf-8")]=emb
        #vocab.append(row[0])
        #embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    
    return emb_dict


def headline_emb_multi_EXT(headlines_list_text, embeddings_dict, ext_features_file):
    ext_features=read_ext_features(ext_features_file)
    print "works multi EXT"
    matrix=[]
    oov={}
    check=len(embeddings_dict["for"])    
    i=0
    for headline in headlines_list_text:
       
        sum_vektor_h=np.random.rand(check)
        sum_vektor_h=np.ones(check)
        for word in headline:
            if word in embeddings_dict.keys(): 
                sum_vektor_h=sum_vektor_h*(np.array(embeddings_dict[word]))
            elif word in oov:
                sum_vektor_h=sum_vektor_h*(oov[word])
            else:
                oov[word]=np.random.rand(len(sum_vektor_h))
                sum_vektor_h=sum_vektor_h*(oov[word])

        #print len(sum_vektor_h)
        if len(sum_vektor_h)==check:
        
            print >> sys.stderr, i , ' von ', len(headlines_list_text)
            matrix.append(sum_vektor_h.tolist()+ext_features[i])
            print ext_features[i]                
        else:
            print "sollte nicht mehr passieren!!"
            matrix.append(np.zeros(check).tolist())
        i=i+1
    return matrix


def headline_emb_multi(headlines_list_text, embeddings_dict):
    print "works"
    matrix=[]
    oov={}
    check=len(embeddings_dict["for"])    
    i=0
    for headline in headlines_list_text:
       
        #sum_vektor_h=np.random.rand(check)
        sum_vektor_h=np.ones(check)
        for word in headline:
            if word in embeddings_dict.keys(): 
                sum_vektor_h=sum_vektor_h*(np.array(embeddings_dict[word]))
            elif word in oov:
                sum_vektor_h=sum_vektor_h*(oov[word])
            else:
                oov[word]=np.random.rand(len(sum_vektor_h))
                sum_vektor_h=sum_vektor_h*(oov[word])

        #print len(sum_vektor_h)
        if len(sum_vektor_h)==check:
        
            print >> sys.stderr, i , ' von ', len(headlines_list_text)
            matrix.append(sum_vektor_h.tolist())                
        else:
            print "sollte nicht mehr passieren!!"
            matrix.append(np.zeros(check).tolist())
        i=i+1
    return matrix


def loadGoogle(filename):
    vocab = []
    embd = []
    emb_dict={}
    i=0
    file = open(filename,'r')
    for line in file.readlines()[1:]:


        row = line.strip().split(' ')
        
        emb=map(float, row[1:])
        emb_dict[row[0].decode("utf-8")]=emb
        #vocab.append(row[0])
        #embd.append(row[1:])
    print('Loaded Google!')
    file.close()
    
    return emb_dict

def read_ext_features(filename):
    features=[]
    f=open(filename, "r")
    for line in f.readlines():
        
        row = line.split(',')
        row=row[1:]
        row=map(str.strip, row)
        row = map(float, row)
        features.append(row)
    f.close()
    return features
