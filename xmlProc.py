from xml.dom.minidom import parse
import xml.dom.minidom

def readXML(train_corpus, test_corpus):
    # Open XML document using minidom parser
    #DOMTree = xml.dom.minidom.parse(new_test_corpus)
    DOMTree_test = xml.dom.minidom.parse(test_corpus)
    #DOMTree = xml.dom.minidom.parse("small_changed_guardian_test.xml")
    collection_test = DOMTree_test.documentElement
    
    #DOMTree = xml.dom.minidom.parse(new_train_corpus)
    DOMTree_train = xml.dom.minidom.parse(train_corpus)
    #DOMTree= xml.dom.minidom.parse("small_changed_guardian_train.xml")
    collection_train = DOMTree_train.documentElement
    # Get all the movies in the collection
    headlines_train = collection_train.getElementsByTagName("headline")
    headlines_test = collection_test.getElementsByTagName("headline")
    
    return headlines_train , headlines_test
