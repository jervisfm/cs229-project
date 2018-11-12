import time

from sklearn.linear_model import LogisticRegression

import massageData

def main():
    dataGetter = massageData.massageData()
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, Y_train)
    print ('Prediction: ', clf.predict(X_dev))
    print ('Actual Label: ', Y_dev)
    print ('Accuracy: ', clf.score(X_dev, Y_dev))
    

if __name__ == '__main__':
    main()