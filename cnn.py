import numpy as np
import pandas
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import TensorBoard

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import massageData
import utils
from sklearn.metrics import confusion_matrix

CLASS_NUM = 4

def encode_values(encoder, Y):
    # Use train y to encode
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y
    # return encoded_Y

def decode_values(encoder, dummy_y):
    # encoded_Y = np.argmax(dummy_y)
    # return encoder.inverse_transform(encoded_Y)
    return encoder.inverse_transform(dummy_y)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1, input_dim=utils.BITMAP_DIM, activation='sigmoid'))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    print ("Done loading the libraries")
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load dataset
    data = massageData.massageData()
    X, Y = data.getTrain()
    X_dev, Y_dev = data.getDev()

    print ("Done load dataset")

    # do some more preprocessing
    # encode class values as integers
    encoder = LabelEncoder()
    # dummy_y will be one-hot encoding of classes
    dummy_y = encode_values(encoder, Y)
    dummy_y_dev = encode_values(encoder, Y_dev)

    print ('Dummy_y (should be one vector if class numbers):', dummy_y)

    print ("Done preprocessing dataset")

    # build the model
    tensorboard = TensorBoard()
    estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=500, verbose=1)
    print ('Y passing into model: ', dummy_y)
    estimator.fit(X, dummy_y)
    dummy_y_pred_dev = estimator.predict(X_dev)

    print ("Dummy y predict (should be one long vector of number): {}".format(dummy_y_pred_dev))

    # print("predictions ", estimator.predict(X_dev))
    # print("actual output ", Y_dev)

    print ("Done building estimator")

    # kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

    # results = cross_val_score(estimator, X, dummy_y, cv=kfold, verbose=1, fit_params={'callbacks': [tensorboard]})
    t1 = time.time()

    print("Time elapsed: ", t1 - t0)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    print ("Confusion matrix", confusion_matrix(Y_dev, decode_values(encoder, dummy_y_pred_dev)))

if __name__ == '__main__':
    main()