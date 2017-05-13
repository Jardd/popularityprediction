import json
import textProcessing





def prep_train_skipgram(train_json):
    with open(train_json) as data_file:    
        train_headlines_list_text= json.load(data_file)
    train_headlines_list_text=textProcessing.getWordsFromJson(train_headlines_list_text)
    train_headlines_list_text=textProcessing.stem(train_headlines_list_text)
    train_headlines_list_text=sum(train_headlines_list_text,[])
    train_headlines_list_text = filter(None, train_headlines_list_text)
    return train_headlines_list_text 

def prep_corpora(train_json, corpus):
    with open(train_json) as data_file:    
        train_headlines_list_text= json.load(data_file)
   # print train_headlines_list_text
   

    train_headlines_list_text=textProcessing.getWordsFromJson(train_headlines_list_text)
    train_headlines_list_text=textProcessing.stem(train_headlines_list_text)
    
    f=open("tf_"+corpus, "w")
    for headline in train_headlines_list_text:
        #print(headline)
        for word in headline:
            #print word
            f.write(word.encode("utf-8"))
            f.write(" ")
	f.write("XYZ XYZ XYZ XYZ ")

    f.close()
