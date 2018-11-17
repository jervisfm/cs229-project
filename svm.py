import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pickle

import massageData
import utils

def kernel_svm(**args):
    print(args)
    dataGetter = massageData.massageData()
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()

    print("Starting Linear SVM training ...")
    clf = svm.SVC(**args)
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
    file_name = "cm_" + args['kernel']
    utils.create_confusion_matrices(class_names, confusion, file_name)

def main():
    kernel_svm(kernel='linear', verbose=1, max_iter=1)
    #kernel_svm(kernel='polynomial', degree=, coef0=, verbose=1)
    #kernel_svm(kernel='rbf', gamma=, verbose=1)
    #kernel_svm(kernel='sigmoid', coef0=, verbose=1)

if __name__ == '__main__':
    main()
