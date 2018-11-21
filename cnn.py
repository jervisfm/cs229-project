import numpy as np
import pandas
import time
import random

import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import massageData
import utils
from sklearn.metrics import confusion_matrix

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_iter', 100, 'Number of steps/epochs to run training.')
flags.DEFINE_string('data_folder', 'data/numpy_bitmap/', 'Directory which has training data to use. Must have / at end.')
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')


CLASS_NUM = 50

model_filename = 'simple_cnn_keras_model'
def encode_values(encoder, Y, forConfusionMatrix=False):
    # Use train y to encode
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    if not forConfusionMatrix:
        # Want 1-hot for training
        dummy_y = np_utils.to_categorical(encoded_Y)
        return dummy_y
    else:
        # Want the class labels (numbers) for confusion.
        return encoded_Y

    # return encoded_Y

def decode_values(encoder, dummy_y):
    # encoded_Y = np.argmax(dummy_y)
    # return encoder.inverse_transform(encoded_Y)
    return encoder.inverse_transform(dummy_y)

def model():
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(14, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(CLASS_NUM, activation='softmax'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

def get_onehot_vector(index, num_classes):
    result = np.zeros(num_classes)
    result[index] = 1
    return result

def convert_predictions_to_onehot(predictions, num_classes):
    num_examples = predictions.shape[0]
    result = np.zeros((num_examples, num_classes))
    for i in range(num_examples):
        result[i] = get_onehot_vector(predictions[i], num_classes)
    return result


def main():
    print ("Done loading the libraries")
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    random.seed(seed)

    # load dataset
    data = massageData.massageData()
    X, Y = data.getTrain()
    X_dev, Y_dev = data.getDev()

    class_names = utils.get_label(Y_dev)
    # Reshape to CNN format (M, 28, 28, 1) from (M, 784)
    print("X shape before:", X.shape)
    X = X.reshape((-1, 28, 28, 1))
    print("X shape after: ", X.shape)

    print("X dev shape before:", X_dev.shape)
    X_dev = X_dev.reshape((-1, 28, 28, 1))
    print("X dev shape after: ", X_dev.shape)

    print ("Done load dataset")

    # do some more preprocessing
    # encode class values as integers
    encoder = LabelEncoder()
    # dummy_y will be one-hot encoding of classes
    dummy_y = encode_values(encoder, Y)
    dummy_y_dev = encode_values(encoder, Y_dev)
    dummy_y_dev_confusion_matrix = encode_values(encoder, Y_dev, forConfusionMatrix=True)

    print ('Dummy_y (should be one vector if class numbers):', dummy_y)

    print ("Done preprocessing dataset")

    # build the model
    tensorboard = TensorBoard()

    cnn_model = model()
    cnn_model.fit(X, dummy_y, epochs=100, batch_size=500, verbose=1)
    dummy_y_pred_dev = cnn_model.predict(X_dev)

    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)

    # evaluate the model
    scores = cnn_model.evaluate(X_dev, dummy_y_dev,  verbose=0)
    print("Cnn model metrics", cnn_model.metrics_names)
    print("%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    # TODO: make this configurable.
    model_json = cnn_model.to_json()
    with open("cnn_model_50.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_model.save_weights("cnn_model_weights_50.keras")
    print("Saved model to disk")



    # print("predictions ", estimator.predict(X_dev))
    # print("actual output ", Y_dev)

    print ("Done building estimator")

    # kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

    # results = cross_val_score(estimator, X, dummy_y, cv=kfold, verbose=1, fit_params={'callbacks': [tensorboard]})
    t1 = time.time()

    print("Time elapsed: ", t1 - t0)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


    print("Dummy y pred dev class", dummy_y_pred_dev_class[0])
    conf_matrix = confusion_matrix(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class)
    print ("Confusion matrix", conf_matrix)
    pickle.dump(class_names, open('cnn_50class', 'wb'))
    pickle.dump(conf_matrix, open('cnn_confusion_matrix_50class', 'wb'))
    
if __name__ == '__main__':
    main()
