import analyse

def headlines_to_textList(headlines):
    headlines_list_text=[]
    for headline in headlines:
        h_text=[]
        text=headline.getElementsByTagName("text")[0].childNodes[0].data.encode('utf-8')
        h_text=text.split()
        headlines_list_text.append(h_text)
    return headlines_list_text
    
def get2ClassesTrain(headlines, tag):
    classes=[]
    threshold=analyse.find_median(analyse.make_list_num(headlines, tag))
    for headline in headlines:
        value=int(headline.getElementsByTagName(tag)[0].childNodes[0].data)
        if value<=threshold:
            classes.append(0)
        else:
            classes.append(1)
    return classes, threshold
    
def get2ClassesTest(headlines, tag, threshold):
    classes=[]
    #threshold=analyse.find_median(analyse.make_list_num(headlines, tag))
    for headline in headlines:
        value=int(headline.getElementsByTagName(tag)[0].childNodes[0].data)
        if value<=threshold:
            classes.append(0)
        else:
            classes.append(1)
    return classes    
def get4ClassesTrain(headlines, tag):
    classes=[]
    zahlen_liste=analyse.make_list_num(headlines, tag)
    threshold2=analyse.find_median(zahlen_liste)
    threshold1=zahlen_liste[((len(zahlen_liste)/2)/2)]
    threshold3=zahlen_liste[(len(zahlen_liste)/2)+((len(zahlen_liste)/2)/2)]
    thresholds=[threshold1, threshold2, threshold3]
    print "t1: ", threshold1
    print "t2: ", threshold2
    print "t3: ", threshold3
    for headline in headlines:
        value=int(headline.getElementsByTagName(tag)[0].childNodes[0].data)
        if value<=threshold1:
            classes.append(0)
        elif value >threshold1 and value <=threshold2:
            classes.append(1)
        elif value >threshold2 and value<=threshold3:
            classes.append(2)
        else:
            classes.append(3)
    return classes, thresholds
    
def get4ClassesTest(headlines, tag, thresholds):
    print thresholds
    classes=[]
    for headline in headlines:
        value=int(headline.getElementsByTagName(tag)[0].childNodes[0].data)
        if value<=thresholds[0]:
            classes.append(0)
        elif value >thresholds[0]and value <=thresholds[1]:
            classes.append(1)
        elif value >thresholds[1] and value<=thresholds[2]:
            classes.append(2)
        else:
            classes.append(3)
    return classes

def headline_vektor_from_features(headlines_list_text, features):
    headlines_vektor_binary=[]
    for headline in headlines_list_text:
        headline_vektor=[0] * len(features)
        for feature in features:
            if feature in headline:
                headline_vektor[features.index(feature)]=1
        headlines_vektor_binary.append(headline_vektor)
    return headlines_vektor_binary

def listToString(headlines_list_text):
    str_headlines=[]
    for headline in headlines_list_text:
        #print headline
        string=" ".join(headline)
        str_headlines.append(string)
    return str_headlines
