import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pickle

import massageData
import utils

def main():
    dataGetter = massageData.massageData()
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()

    print("Starting Linear SVM training ...")
    clf = svm.LinearSVC(verbose=1)
    clf.fit(X_train, Y_train)
    print("Trained")

    Y_dev_prediction = clf.predict(X_dev)
    print("Predictions")
    print ('Prediction: ', Y_dev_prediction)
    print ('Actual Label: ', Y_dev)
    print ('Accuracy: ', clf.score(X_dev, Y_dev))

    class_names = utils.get_label(Y_dev)
    confusion = confusion_matrix(Y_dev, Y_dev_prediction, labels=class_names)
    print ("Confusion matrix: ", confusion)
    pickle.dump(class_names, open("class_names_svm_l", 'wb'))
    pickle.dump(confusion, open("confusion_matrix_nclass_svm_l", 'wb'))
    utils.create_confusion_matrices(class_names, confusion)

if __name__ == '__main__':
    main()