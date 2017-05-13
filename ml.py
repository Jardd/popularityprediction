

import analyse
from sklearn import svm

def fit_training_svm(all_vektors, all_classes):
    clf = svm.SVC()
    #all_vektors=np.array(all_vektors)
    #all_classes=np.array(all_classes)
    clf.fit(all_vektors, all_classes) 
    return clf
    
