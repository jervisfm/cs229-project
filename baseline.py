import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from massageData import massageData
# TODO: make split (train, test, dev)
CLASS_NUM = 3

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1, input_dim=massageData.BITMAP_DIM, activation='sigmoid'))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    print ("Done loading the libraries")
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)



    # load dataset
    data = massageData()
    X = data.getX()
    Y = data.getY()
    print ("Done load dataset")

    #TODO: from 3 files, create the label and combine to 1 training data file

    # do some more preprocessing
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    print ("Done preprocessing dataset")

    # build the model
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

    print ("Done building estimator")

    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

if __name__ == '__main__':
    main()