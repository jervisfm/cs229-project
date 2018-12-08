import numpy as np
import pandas
import time
import random
import os

import scipy

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

#TODO: consider scaling image to 299x299.

from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_iter', 100, 'Number of steps/epochs to run training.')
flags.DEFINE_integer('batch_size', 500, 'Number of examples to use in a batch for stochastic gradient descent.')
flags.DEFINE_string('data_folder', 'data/numpy_bitmap/', 'Directory which has training data to use. Must have / at end.')
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')
flags.DEFINE_integer('model_version', 1, 'The version of the model that we want to run. Useful to run different models to compare them')
flags.DEFINE_boolean('using_binarization', False, 'If True, binarize the data before passing into cnn')


model_filename = 'transfer_learning_model'
model_weights_filename = 'transfer_learning_model_weights'
class_file_name = 'class_names_transfer_learning'
confusion_file_name = 'confusion_matrix_transfer_learning'

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
    filename  = "{}{}".format(model_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_model_weights_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(model_weights_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename =  "{}{}".format("transfer_learning_keras_results", suffix_name)
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


def transfer_learning(X, y, source_model=InceptionV3(weights='imagenet', include_top=False)):
    # Based on code snippets from:https://keras.io/applications/
    print("Running transfer learning ...")
    # create the base pre-trained model
    base_model = source_model

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # Let's reduce to 128 as that's what gave us good perf in the simple model.
    x = Dense(128, activation='relu')(x)
    #x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(get_num_classes(), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    print("Tuning our last custom layer...")
    model.fit(X, y, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("Tuning the last 2 inceptions layers ...")
    model.fit(X, y, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size, verbose=1)
    return  model

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


def convertTo3Channels(x, new_image_size=(299, 299)):
        # x has shape (m. size, size, # channels)
        num_examples, size_x, size_y, _ = x.shape
        result = np.zeros((num_examples, new_image_size[0], new_image_size[1], 3))
        num_channels = 3
        # TODO: Try red color drawings...
        for i in range(num_examples):
            source_image = x[i, :, :, 0]
            scaled_image = scipy.misc.imresize(source_image, new_image_size)
            for j in range(num_channels):
                result[i, :, :, j] = scaled_image

        return result


def assertXIsNotNan(X):
    if np.isnan(X).any():
        raise ValueError("Oh no input is nan!!!")


def main():
    print("folder = ", FLAGS.data_folder)
    t0 = time.time()
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    random.seed(seed)

    # load dataset
    # TODO: make this configurable.
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

    print("Converting to 3 channnels...")
    
    print ("Done load dataset")

    # do some more preprocessing    # encode class values as integers
    encoder = LabelEncoder()
    # dummy_y will be one-hot encoding of classes
    dummy_y = encode_values(encoder, Y)
    dummy_y_dev = encode_values(encoder, Y_dev)
    dummy_y_dev_confusion_matrix = encode_values(encoder, Y_dev, forConfusionMatrix=True)

    print ('Dummy_y (should be one vector if class numbers):', dummy_y)

    print("Converting input images for transfer learning...")
    # Image size expected for transfer learned model.
    # Inceptionv3 -- 299x299
    # Resnet -- 224 x 224

    #new_image_size = (299, 299)
    new_image_size = (224, 224)
    X_dev = convertTo3Channels(X_dev, new_image_size)
    X = convertTo3Channels(X, new_image_size)
    
    print ("Done preprocessing dataset")

    # build the model
    tensorboard = TensorBoard()

    # TODO: make this configurabel via flag.
    source_model = MobileNet(weights='imagenet', include_top=False)
    #source_model = InceptionV3(weights='imagenet', include_top=False)
    #source_model = ResNet50(weights='imagenet', include_top=False)
    model = transfer_learning(X, dummy_y, source_model)

    t1 = time.time()
    training_duration_secs = t1 - t0
    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format( 1.0 * training_duration_secs / FLAGS.max_iter)

    dummy_y_pred_dev = model.predict(X_dev)

    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)


    # evaluate the model
    scores = model.evaluate(X_dev, dummy_y_dev,  verbose=0)
    print("Model metric names: ", model.metrics_names)
    experiment_result_string += "Tranfer learning model result  %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100)

    print(experiment_result_string)


def generate_confusion_matrix():
    class_names = pickle.load(open(get_class_filename(), 'rb'))[:10]
    confusion = pickle.load(open(get_confusion_matrix_filename(), 'rb'))[:10,:10]
    utils.create_confusion_matrices(class_names, confusion, get_confusion_matrix_name())

if __name__ == '__main__':
    main()
    #generate_confusion_matrix()
