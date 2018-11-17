import time
import itertools
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_string('data_folder', 'data/numpy_bitmap/', 'Directory which has training data to use')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')

import massageData
import utils

class_file_name = 'class_names_baselinev2_lr'
confusion_file_name = 'confusion_matrix_baselinev2_lr'

def get_suffix_name():
    return "_" + FLAGS.experiment if FLAGS.experiment_name else ""

def get_class_filename():
    suffix_name = get_suffix_name()
    return "{}{}".format(class_file_name, suffix_name)

def get_confusion_matrix_filename():
    suffix_name = get_suffix_name()
    return "{}{}".format(confusion_file_name, suffix_name)

def run():
    print("folder = ", FLAGS.data_folder)

    dataGetter = massageData.massageData(folder=FLAGS.data_folder)

    X_train, Y_train = dataGetter.getTrain() #TODO: feature extractions
    X_dev, Y_dev = dataGetter.getDev()
    start_time_secs = time.time()
    print("Starting Logistic Regression training ...")
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', verbose=10, n_jobs=-1).fit(X_train, Y_train)
    print("Training done.")
    end_time_secs = time.time()
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
    pickle.dump(class_names, open(get_class_filename(), 'wb'))
    pickle.dump(confusion, open(get_confusion_matrix_filename(), 'wb'))
    
def main():
    class_names = pickle.load(open(get_class_filename(), 'rb'))[:10]
    confusion = pickle.load(open(get_confusion_matrix_filename(), 'rb'))[:10,:10]
    utils.create_confusion_matrices(class_names, confusion)




if __name__ == '__main__':
    run()
    #main()
