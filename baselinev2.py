import time
import itertools
import numpy as np
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_iter', 100, 'Number of steps to run trainer.')
flags.DEFINE_string('data_folder', 'data/numpy_bitmap/', 'Directory which has training data to use. Must have / at end.')
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')

import massageData
import utils

class_file_name = 'class_names_baselinev2_lr'
confusion_file_name = 'confusion_matrix_baselinev2_lr'

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

def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename =  "{}{}".format("baselinev2_lr_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)


def run():
    print("folder = ", FLAGS.data_folder)

    dataGetter = massageData.massageData(folder=FLAGS.data_folder)

    X_train, Y_train = dataGetter.getTrain() #TODO: feature extractions
    X_dev, Y_dev = dataGetter.getDev()
    start_time_secs = time.time()
    print("Starting Logistic Regression training ...")
    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             multi_class='multinomial',
                             verbose=10,
                             n_jobs=-1,
                             max_iter=FLAGS.max_iter).fit(X_train, Y_train)
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
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format( 1.0 * training_duration_secs / FLAGS.max_iter)

    class_names = utils.get_label(Y_dev)
    classification_report_string = classification_report(Y_dev, Y_dev_prediction, target_names=class_names)
    experiment_result_string += "\nClassification report: {}".format(classification_report_string)

    print(experiment_result_string)

    # Save report to file
    write_contents_to_file(get_experiment_report_filename(), experiment_result_string)
    confusion = confusion_matrix(Y_dev, Y_dev_prediction, labels=class_names)

    print ("Confusion matrix: ", confusion)
    pickle.dump(class_names, open(get_class_filename(), 'wb'))
    pickle.dump(confusion, open(get_confusion_matrix_filename(), 'wb'))
    
def main():
    class_names = pickle.load(open(get_class_filename(), 'rb'))[:10]
    confusion = pickle.load(open(get_confusion_matrix_filename(), 'rb'))[:10,:10]
    utils.create_confusion_matrices(class_names, confusion, get_confusion_matrix_name())




if __name__ == '__main__':
    run()
    #main()
