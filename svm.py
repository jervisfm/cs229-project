import time
from sklearn import svm
from sklearn.metrics import confusion_matrix

import massageData


def main():
	dataGetter = massageData.massageData()
	X_train, Y_train = dataGetter.getTrain()
	print("Got training data")
	X_dev, Y_dev = dataGetter.getDev()
	print("Got dev data")

	clf = svm.SVC(verbose=1)
	clf.fit(X_train, Y_train)
	print("Trained")

	Y_dev_prediction = clf.predict(X_dev)
	print("Predictions")
	print ('Prediction: ', Y_dev_prediction)
	print ('Actual Label: ', Y_dev)
	print ('Accuracy: ', clf.score(X_dev, Y_dev))
	print ("Confusion matrix: ", confusion_matrix(Y_dev, Y_dev_prediction))

if __name__ == '__main__':
	main()