import numpy as np
import pandas
import time
import random
import os
import keras
import datetime

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
from training_plot import TrainingPlot


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
flags.DEFINE_integer('batch_size', 500, 'Number of examples to use in a batch for stochastic gradient descent.')
flags.DEFINE_string('data_folder', 'data/numpy_bitmap/', 'Directory which has training data to use. Must have / at end.')
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')
flags.DEFINE_integer('model_version', 1, 'The version of the model that we want to run. Useful to run different models to compare them')
flags.DEFINE_boolean('using_binarization', False, 'If True, binarize the data before passing into cnn')


model_filename = 'simple_cnn_keras_model'
model_weights_filename = 'simple_cnn_keras_model_weights'
class_file_name = 'class_names_simple_cnn'
confusion_file_name = 'confusion_matrix_simple_cnn'

def get_data_folder():
    return FLAGS.data_folder

def get_num_classes():
    return len(os.listdir(get_data_folder()))

def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""

def get_class_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(class_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_confusion_matrix_name():
    return "{}{}".format(confusion_file_name, get_suffix_name())

def get_confusion_matrix_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(confusion_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_model_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_tensorboard_directory():
    tensorboard_root_folder = os.path.join(FLAGS.results_folder, "tensorboard/")
    if not os.path.exists(tensorboard_root_folder):
        os.mkdir(tensorboard_root_folder)

    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    dirpath = os.path.join(tensorboard_root_folder, filename) + "/"

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    now = datetime.datetime.now()
    now_timestring = now.strftime("%Y%m%d_%H%M%S")

    timestamped_dirpath = os.path.join(dirpath, now_timestring)
    if not os.path.exists(timestamped_dirpath):
        os.mkdir(timestamped_dirpath)

    return timestamped_dirpath

def get_training_plots_directory():
    root_folder = os.path.join(FLAGS.results_folder, "training_plots/")
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    dirpath = os.path.join(root_folder, filename) + "/"

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath

def get_model_name_only():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    return filename

def get_training_plot_filename():
    directory = get_training_plots_directory()
    return os.path.join(directory, "{}_training_plot_loss.png".format(get_model_name_only()))

def get_tensorboard_callback(frequency=2):
    logdir = "./" + get_tensorboard_directory()
    print("Tensorboard logdir: ", logdir)
    return keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=frequency,
    write_graph=True, write_images=True)

def get_model_weights_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(model_weights_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename =  "{}{}".format("simple_cnn_keras_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


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
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

def model_v2():
    # model_v2() only has 1 conv layer compared to 2 conv layers in model()
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 2')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # # Adding a second convolutional layer
    # classifier.add(Conv2D(14, (3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier    

def model_v3():
    # model_v3() only has 1 conv layer compared to 2 conv layers in model() and removed max pool
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 3')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier    

def model_v4():
    # model_v2() only has 1 conv layer compared to 2 conv layers in model()
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 2 dense 256')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # # Adding a second convolutional layer
    # classifier.add(Conv2D(14, (3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
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
    print("folder = ", FLAGS.data_folder)
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    random.seed(seed)



    # load dataset
    data = massageData.massageData(folder=FLAGS.data_folder, binarize=FLAGS.using_binarization)
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

    
    # add flags to switch model
    if FLAGS.model_version == 2:
        cnn_model = model_v2()
    if FLAGS.model_version == 3:
        cnn_model = model_v3()
    if FLAGS.model_version == 4:
        cnn_model = model_v4()
    else:
        cnn_model = model()

    plot_losses = TrainingPlot(get_training_plot_filename())
    cnn_model.fit(X, dummy_y, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1, callbacks=[get_tensorboard_callback(), plot_losses], validation_data=(X_dev, dummy_y_dev))

    t1 = time.time()
    training_duration_secs = t1 - t0
    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format( 1.0 * training_duration_secs / FLAGS.max_iter)

    dummy_y_pred_dev = cnn_model.predict(X_dev)

    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)


    # evaluate the model
    scores = cnn_model.evaluate(X_dev, dummy_y_dev,  verbose=0)
    experiment_result_string += "Simple CNN model %s: %.2f%%" % (cnn_model.metrics_names[1], scores[1] * 100)

    print(experiment_result_string)
    utils.write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

    # serialize model to JSON
    # TODO: make this configurable.
    model_json = cnn_model.to_json()
    with open(get_model_filename(), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_model.save_weights(get_model_weights_filename())
    print("Saved model to disk")



    # print("predictions ", estimator.predict(X_dev))
    # print("actual output ", Y_dev)

    print ("Done building estimator")

    # kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

    # results = cross_val_score(estimator, X, dummy_y, cv=kfold, verbose=1, fit_params={'callbacks': [tensorboard]})
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


    print("Dummy y pred dev class", dummy_y_pred_dev_class[0])
    conf_matrix = confusion_matrix(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class)
    print ("Confusion matrix", conf_matrix)
    pickle.dump(class_names, open(get_class_filename(), 'wb'))
    pickle.dump(conf_matrix, open(get_confusion_matrix_filename(), 'wb'))

def generate_confusion_matrix():
    class_names = pickle.load(open(get_class_filename(), 'rb'))[:10]
    confusion = pickle.load(open(get_confusion_matrix_filename(), 'rb'))[:10,:10]
    utils.create_confusion_matrices(class_names, confusion, get_confusion_matrix_name())

if __name__ == '__main__':
    main()
    generate_confusion_matrix()
