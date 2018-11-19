import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle

import massageData
import utils

def kernel_svm(**args):
    print(args)
    dataGetter = massageData.massageData()
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()
    
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_dev = scaling.transform(X_dev)

    print("Starting " + args['kernel'] + " SVM training ...")
    start_time_secs = time.time()
    clf = svm.SVC(**args)
    clf.fit(X_train, Y_train)
    end_time_secs = time.time()
    print("Trained")

    training_duration_secs = end_time_secs - start_time_secs
    Y_dev_prediction = clf.predict(X_dev)
    
    accuracy = clf.score(X_dev, Y_dev)
    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nPrediction: {}".format(Y_dev_prediction)
    experiment_result_string += "\nActual Label: {}".format(Y_dev)
    experiment_result_string += "\nAcurracy: {}".format(accuracy)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    print(experiment_result_string)

    class_names = utils.get_label(Y_dev)
    confusion = confusion_matrix(Y_dev, Y_dev_prediction, labels=class_names)
    print ("Confusion matrix: ", confusion)
    pickle.dump(class_names, open("class_names_svm_l", 'wb'))
    pickle.dump(confusion, open("confusion_matrix_nclass_svm_l", 'wb'))
    file_name = "cm_svm_10_" + args['kernel']
    utils.create_confusion_matrices(class_names, confusion, file_name)

def main():
    kernel_svm(kernel='linear', verbose=1, max_iter=500)
    kernel_svm(kernel='poly', degree=5, coef0=1, verbose=1, max_iter=500)
    kernel_svm(kernel='rbf', gamma=1, verbose=1, max_iter=500)
    kernel_svm(kernel='sigmoid', coef0=1, verbose=1, max_iter=500)

if __name__ == '__main__':
    main()
