import json
import textProcessing
import numpy as np

#k =word window
#extrakt skipgrams from train and test corpus, json files already include the NOT- stemed headlines
#returns dict with word and skipgrams
def getSkipgrams(train_corpus, test_corpus, k):
	with open(train_corpus+'.json') as data_file:    
		train_headlines_list_text= json.load(data_file)
	with open(test_corpus+'.json') as data_file:
		test_headlines_list_text= json.load(data_file)
	
	print "test1"
	train_headlines_list_text=textProcessing.getWordsFromJson(train_headlines_list_text)
	train_headlines_list_text=textProcessing.stem(train_headlines_list_text)
	
	test_headlines_list_text=textProcessing.getWordsFromJson(test_headlines_list_text)
	test_headlines_list_text=textProcessing.stem(test_headlines_list_text)
	print "test2"
	
	train_headlines_list_text=[train_headlines_list_text[0]]
	skipgrams={}
	for headline in train_headlines_list_text:
		#print headline
		headline=filter(None, headline)
		for w in range(0,len(headline)):
			skg=[]
			#print w
			#get word window after word
			n=1
			while n <= k and (n+w)<len(headline):
				skg.append( headline[w+n])
				n+=1
			#print skg

			#get word window before word	
			#print "***second rotation****"
			n=k
			while n > 0 and (w-n)>=0:
				skg.append(headline[w-n])
				n-=1
			#print skg
			word=headline[w]
			if word not in skipgrams:
				skipgrams[word]=skg
			else:
				skipgrams[word]=skipgrams[word]+skg
				
			#print word
			#print skipgrams
	#print len(skg)
	#skg=set(skg)
	#print len(skg)
	#print skg
	#print "test3"
	print skipgrams
	return skipgrams





#skipgrams is a dict with inputwords x and there contextwords (y1, y2..)
def ready_input(skipgrams):

	vocabulary=skipgrams.keys()
	print vocabulary
	skipgramsInt={}
	for key in skipgrams:
		keyInt=vocabulary.index(key)	
		#keyInt=np.eye(len(vocabulary),dtype=int)[[keyInt]]
		#keyInt=tuple(keyInt[0])
		skipgramsInt[keyInt]=[]
		for value in skipgrams[key]:
			valueInt=vocabulary.index(value)
			skipgramsInt[keyInt].append(valueInt)
		#skipgramsInt[keyInt]=np.eye(len(vocabulary),dtype=int)[skipgramsInt[keyInt]]
	print skipgramsInt
if (__name__=="__main__"):
	input_data=getSkipgrams("corpora/changed_guardian_train.xml", "corpora/changed_guardian_test.xml", 2)
	ready_input(input_data)








