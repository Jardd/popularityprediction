import sys
import xmlProc
import json
import vektorMaker
import textProcessing
import analyse
import ml
import featureSelection
train_corpus=str(sys.argv[1])
test_corpus=str(sys.argv[2])
split=int(sys.argv[3])
tag=str(sys.argv[4])
ext_feature_file_train=str(sys.argv[5])
ext_feature_file_test=str(sys.argv[6])

headlines_train , headlines_test =xmlProc.readXML(train_corpus, test_corpus)

with open(train_corpus+'.json') as data_file:    
    train_headlines_list = json.load(data_file)
with open(test_corpus+'.json') as data_file:
    test_headlines_list = json.load(data_file)

      
if split ==2:
    train_classes, threshold=vektorMaker.get2ClassesTrain(headlines_train, tag)
    test_classes=vektorMaker.get2ClassesTest(headlines_test, tag,threshold)
elif split ==4:
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
    #print train_headlines_list_text

global oov
oov={}
print "start making train_emb"
i=0

train_matrix=[]
f=open(ext_feature_file_train, "r")
for line in f.readlines():
        
    row = line.split(',')
    row=row[1:]
    row=map(str.strip, row)
    row = map(float, row)
    train_matrix.append(row)
f.close()

print "train to emb finished"

print len(train_matrix)
print "stat making test_emb"

test_matrix=[]
f2=open(ext_feature_file_test, "r")
for line in f2.readlines():
        
    row = line.split(',')
    row=row[1:]
    row=map(str.strip, row)
    row = map(float, row)
    test_matrix.append(row)
f2.close()

train_matrix, test_matrix=featureSelection.MI(train_matrix, test_matrix, train_classes, 100)


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

