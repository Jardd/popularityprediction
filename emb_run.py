# -*- coding: utf-8 -*-

import sys
import run


if ( __name__ == "__main__"):
    print "start"
    #mein.py training.xml test.xml split tag featueselection wertFuerFS
    # main.py "corpora\changed_guardian_train.xml" "corpora\changed_guardian_test.xml" 2 T1 SVC 125
    #main.py small_changed_guardian_train.xml small_changed_guardian_test.xml 2 T1

    train_corpus=str(sys.argv[1])
    test_corpus=str(sys.argv[2])
    split=int(sys.argv[3])
    tag=str(sys.argv[4])
    emb=str(sys.argv[5])
    weighting=str(sys.argv[6])
    print "sdfds"
    run.skipgram(train_corpus, test_corpus, split, tag, emb, weighting)

