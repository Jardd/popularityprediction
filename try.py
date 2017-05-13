import baseline
import featureSelection
import nltk
#from sklearn import svm
#from sklearn.svm import LinearSVC
#from sklearn.feature_selection import SelectFromModel

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
"""
s1="That is an example"
s2="This is another example"
s3="Something totally different"
s4="more stuff"

"that, is, an ,example , this, another something, totally, different, more ,stuff"
vekt_train=[[1,1,1,1,0,0,0,0,0,0,0],[0,1,0,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0], [0,1,0,0,0,1,0,1,1,0,0]]
cl=[1,1,0,0]

t="What is this example"

vekt_test=[[0,1,0,1,1,0,0,0,0,0,0]]



lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(vekt_train, cl)
model = SelectFromModel(lsvc, prefit=True)
train_new = model.transform(vekt_train)
print train_new.shape
print train_new

test_new = model.transform(vekt_test)
print test_new

clf = svm.SVC()
clf.fit(train_new, cl)
res=clf.predict(test_new)
print res"""
headlines_test_text=[["Das", "ist" , "nummer", "eins"],["Das" , "ist" , "nummer", "zwei"],["Das", "wird", "was", "ganz", "anderes"]]
#classes=[0,0,1]
#featureSelection.PMIfeature_seletion(classes, headlines_test_text)
headlines=["That's an example!", "This is another one, the 2nd one"]

lemmatizer = nltk.stem.WordNetLemmatizer()

#print(lemmatizer.lemmatize("better", pos="a"))



new_headlines=[]
for headline in headlines:
        text = nltk.word_tokenize(headline)
        text_pos=nltk.pos_tag(text)
        new_headlines.append(text_pos)
        
lemmatizer = nltk.stem.WordNetLemmatizer()

#print lemmatizer.lemmatize("That", "DT")


for headline in new_headlines:
    for word_pos in headline:
        #print word_pos[0]+"  "+word_pos[1]
        pos=get_wordnet_pos(word_pos[1])
        if pos != "":
            print lemmatizer.lemmatize(word_pos[0], pos)
        else:
            print word_pos[0]

