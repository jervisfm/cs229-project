import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import massageData

def main():
    dataGetter = massageData.massageData()
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, Y_train)
    Y_dev_prediction = clf.predict(X_dev)
    print ('Prediction: ', Y_dev_prediction)
    print ('Actual Label: ', Y_dev)
    print ('Accuracy: ', clf.score(X_dev, Y_dev))
    print ("Confusion matrix: ", confusion_matrix(Y_dev, Y_dev_prediction))


if __name__ == '__main__':
    main()