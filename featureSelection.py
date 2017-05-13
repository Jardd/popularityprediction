import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from collections import defaultdict 
import math
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import numpy
"""FeatureSelection"""

def brownFS(brown_cluster, headlines_list_text, valueVektor):
    i=0
    featureVektor=[]
    #valueVektor=[]
    for key in brown_cluster:
        i+=1
        featureVektor.append(key)
        #valueVektor.append(brown_cluster[key])
    print "number clusters: ", i
    print len(featureVektor) , " : ", len(valueVektor)
    number_headlines=len(headlines_list_text)
    matrix=numpy.zeros((number_headlines, len(valueVektor)))
    #print "************************"
    for x in range(0 ,len(headlines_list_text)):
        #print headlines_list_text[x], "\n"
        for y in range(0,len(valueVektor)):
     #       print valueVektor[y]
            #print train_headlines_list_text[x]
            #print valueVektor[y]
            
            if len(set(headlines_list_text[x]).intersection(valueVektor[y])) >0: #have a common word
                matrix[x,y]=1
      #  print "********************"
    print matrix
    print matrix.shape
    
    
    return matrix
    
def MI(train_binary, test_binary, classes_train ,k ):
    mi = SelectKBest(mutual_info_classif, k)
    #vectorizer = CountVectorizer()
    #X = vectorizer.fit_transform(train_headlines_list_text)
    X_train = mi.fit_transform(train_binary, classes_train)
    X_test=mi.transform(test_binary)
    return X_train, X_test
    #return features



def chiSquare(train_binary, test_binary, classes_train, k, features):
    ch2 = SelectKBest(chi2, k)
    #vectorizer = CountVectorizer()
    #X = vectorizer.fit_transform(train_headlines_list_text)
    X_train = ch2.fit_transform(train_binary, classes_train)
    #topf=np.asarray(features)[ch2.get_support()]
    #print topf
    X_test=ch2.transform(test_binary)
    return X_train, X_test
    #return features
    
def all_features(headlines_train_list):
    features=set()
    for headline in headlines_train_list:
        for word in headline:
            features.add(word.lower())
    return list(features)
    

    
def SVCfeature_seletion(train_vektors, test_vektors, classes):
    lsvc = LinearSVC(C=0.1, penalty="l1", dual=False).fit(train_vektors, classes)
    model = SelectFromModel(lsvc, prefit=True)
    train_new = model.transform(train_vektors)
    test_new = model.transform(test_vektors)
    return train_new, test_new
    
def term_document_frequency(train_vektors, noDub):
    freq={}
    for word in noDub:
        f=sum(x.count(word) for x in train_vektors)
        freq[word]=f
    new_features=[]
    for key in freq:
#        print freq[key]
        if freq[key]<=100 and freq[key]>10:
#            print "in if"
            new_features.append(key)
    print new_features
    return new_features

def PMI(classes, headlines_train_list):
    features=set()
    #counting all neccecery data
    n_class0=classes.count(0)
    n_class1=classes.count(1)
    
    occ_counter_0=defaultdict(int)
    occ_counter_1=defaultdict(int)
    terms=set()
    for n in range(0,len(headlines_train_list)):
        for word in headlines_train_list[n]:
            terms.add(word)
            if classes[n]==0:
                occ_counter_0[word]+=1
            elif classes[n]==1:
                occ_counter_1[word]+=1
            else:
                print "Error, invalide klasse"
    #print occ_counter_0
    #print occ_counter_1
    terms=list(terms)
    
    #calculatin PMI### t=term, c=class
    #A=t and c co-occur
    #B= t occurs without c
    #D=c occurs without t
    #N number of documents in c , n_class0=classes.count(0)
    term_pmi_0={}
    term_pmi_1={}
    #for class 0
    for t in terms:
        a=occ_counter_0[t]
        b=occ_counter_1[t]
        #print "b=", b
        d=0
        for key in occ_counter_0:
            if key != t:
                d+=occ_counter_0[t]        
        if a==0:#term only occures in one class==perfect!
            #print "a:", a
            term_pmi_0[t]=-np.inf
            features.add(t)
        else:
            tmp=(a*n_class0)/float(((a+d)*(a+b)))
            pmi=math.log(tmp)
            term_pmi_0[t]=pmi
    #for class 1
    for t in terms:
        a=occ_counter_1[t]
        b=occ_counter_0[t]
        d=0
        for key in occ_counter_1:
            if key != t:
                d+=occ_counter_1[t]        
        if a==0:
            term_pmi_1[t]=-np.inf
            features.add(t)
        else:
            tmp=(a*n_class1)/float(((a+d)*(a+b)))
            pmi=math.log(tmp)
            term_pmi_1[t]=pmi
    
    

    for key in term_pmi_0:
        #print term_pmi_0[key]
        if term_pmi_0[key]<-0.7: 
            #print term_pmi_0[key]
            features.add(key)
    for key in term_pmi_1:
        if term_pmi_1[key]<-0.7: 
            #print term_pmi_1[key]
            features.add(key)
    #print term_pmi_0
    #print term_pmi_1
    #print features
    return list(features)


def PMI2(classes, headlines_train_list, threshold):
    #fuer classe0
    #PMI(term, cl0)=log((#of doc containing term with class cl0 * #doc in train set)\(#of doc containing term *vof doc with cl0))
    ###Notation
    #nc=number of doc with cl0
    #nt=number of doc with term
    #n=number of doc in train set
    #ntc=number of doc with term and cl0
    terms=set()
    features=set()
    pmi_0=defaultdict(float)
    pmi_1=defaultdict(float)
    n=len(classes) #number of doc in train set
    nc0=0
    nc1=0
    terms0=defaultdict(int)
    terms1=defaultdict(int)
    for i in range(0,len(headlines_train_list)): #itterate through docs
        doc=headlines_train_list[i]
        if classes[i]==0:
            nc0=nc0+1
        elif classes[i]==1:
            nc1=nc1+1
        else:
            "Error: mehr als 2 classen?(in pmi2)"
        
        tmp=set() #so no word will be counted twice in same doc
        for word in doc:
            terms.add(word)
            if classes[i]==0 and word not in tmp:
                terms0[word]+=1
            elif classes[i]==1 and word not in tmp:
                terms1[word]+=1
            tmp.add(word)
    
    for term in terms:
        nt=0
        ntc0=0
        ntc1=0
        if term in terms0:
            nt+=terms0[term]
            ntc0+=terms0[term]
        if term in terms1:
            nt+=terms1[term]
            ntc1+=terms0[term]
        if nt==0:
            "Error: in pmi nt"
        
        tmp_pmi_0=(ntc0*n)/(float(nt)*nc0)
        if tmp_pmi_0 != 0.0:
            term_pmi_0=math.log(tmp_pmi_0,2)
            pmi_0[term]=term_pmi_0
        else:
            pmi_0[term]=0

        tmp_pmi_1=(ntc1*n)/(float(nt)*nc1)
        if tmp_pmi_1 != 0.0:
            term_pmi_1=math.log(tmp_pmi_1,2)
            pmi_1[term]=term_pmi_1
        else:
            pmi_1[term]=0
            
    for key in pmi_0:
        if pmi_0[key]>threshold:
            features.add(key)
            #print pmi_0[key]
    for key in pmi_1:
        if pmi_1[key]>threshold:
            features.add(key)
            #print pmi_1[key]
    return list(features)
