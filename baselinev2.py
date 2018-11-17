import time
import itertools
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import massageData
import utils

def run():
    dataGetter = massageData.massageData()
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()
    print("Starting Logistic Regression training ...")
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', verbose=10, n_jobs=-1).fit(X_train, Y_train)
    print("Training done.")
    Y_dev_prediction = clf.predict(X_dev)
    print ('Prediction: ', Y_dev_prediction)
    print ('Actual Label: ', Y_dev)
    print ('Accuracy: ', clf.score(X_dev, Y_dev))
    class_names = utils.get_label(Y_dev)
    confusion = confusion_matrix(Y_dev, Y_dev_prediction, labels=class_names)
    print ("Confusion matrix: ", confusion)
    pickle.dump(class_names, open("class_names_lr", 'wb'))
    pickle.dump(confusion, open("confusion_matrix_nclass_lr", 'wb'))
    
def main():
    class_names = pickle.load(open("class_names_lr", 'rb'))
    confusion = pickle.load(open("confusion_matrix_nclass_lr", 'rb'))
    utils.create_confusion_matrices(class_names, confusion)

#TODO: want to scale this up to 10 classes: add more file, compare runtime


if __name__ == '__main__':
    run()
    main()
