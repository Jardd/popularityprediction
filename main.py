# -*- coding: utf-8 -*-

import sys
import run


if ( __name__ == "__main__"):
    print "start"
    #mein.py training.xml test.xml split tag featueselection wertFuerFS
    # main.py "corpora\changed_guardian_train.xml" "corpora\changed_guardian_test.xml" 2 T1 SVC 125
    #main.py small_changed_guardian_train.xml small_changed_guardian_test.xml 2 T1
    train_corpus=str(sys.argv[1])
    
    if train_corpus=="setUp":
        print train_corpus
        in_file=str(sys.argv[2])
        #run.brownSetUpBE(in_file)
    else:
        
        test_corpus=str(sys.argv[2])
        split=int(sys.argv[3])
        tag=str(sys.argv[4])
        fs=str(sys.argv[5]).split()
        k=int(sys.argv[6])
        print "sdfds"
        run.unigramBaseline(train_corpus, test_corpus, split, tag, fs, k)
        #run.bigramBaseline(train_corpus, test_corpus, split, tag, fs, k)
        
        #run.brown("brown\\brown_output.txt", 400 , train_corpus, test_corpus ,tag)
        #run.skipgram(train_corpus, test_corpus, split, tag, "embeddings/emb_s128_w2_n4.json")

