import os
import pickle
from random import shuffle

import numpy as np

from data_utils import pickle_dump, pickle_load

def get_training_and_validation_generators(training_data_file, validation_data_file, batch_size, training_keys_file, validation_keys_file, train_test_split=0.8, overwrite=False):

    if validation_data_file:
        training_generator = data_generator(training_data_file, range(training_data_file.root.data.shape[0]), batch_size=batch_size)
        validation_generator = val_data_generator(training_data_file, range(validation_data_file.root.data.shape[0]), batch_size=1)
        print 'BATCH SIZE CHECK', batch_size, training_data_file.root.data.shape[0]
        num_training_steps = training_data_file.root.data.shape[0] // batch_size
        num_validation_steps = validation_data_file.root.data.shape[0]
    else:
        training_list, validation_list = get_validation_split(training_data_file, data_split=train_test_split, overwrite=overwrite, training_file=training_keys_file, testing_file=validation_keys_file)
        training_generator = data_generator(training_data_file, training_list, batch_size=batch_size)
        validation_generator = data_generator(training_data_file, validation_list, batch_size=1)
        num_training_steps = len(training_list) // batch_size
        num_validation_steps = len(validation_list)    

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_validation_split(data_file, training_file, testing_file, data_split=0.8, overwrite=False):

    if overwrite or not os.path.exists(training_file):

        num_cases = data_file.root.data.shape[0]
        sample_list = list(range(num_cases))

        training_list, testing_list = split_list(sample_list, split=data_split)

        pickle_dump(training_list, training_file)
        pickle_dump(testing_list, testing_file)

        return training_list, testing_list

    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(testing_file)


def split_list(input_list, split=0.8, shuffle_list=True):

    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1):

    """ TODO: Investigate how generators even work?! And yield.
    """

    while True:
        x_list = []
        y_list = []
        shuffle(index_list)
        for index in index_list:

            add_data(x_list, y_list, data_file, index)

            if len(x_list) == batch_size:

                yield convert_data(x_list, y_list)
                x_list = []
                y_list = []

def val_data_generator(data_file, index_list, batch_size=1):

    """ TODO: Investigate how generators even work?! And yield.
    """

    print 'GENERATED!'

    while True:
        x_list = []
        y_list = []
        shuffle(index_list)
        for index in index_list:

            add_data(x_list, y_list, data_file, index)

            # print 'VALIDATION', len(x_list), index, batch_size

            if len(x_list) == batch_size:

                yield convert_data(x_list, y_list)
                x_list = []
                y_list = []



def add_data(x_list, y_list, data_file, index, augment=False):

    data = data_file.root.data[index]
    truth = data_file.root.truth[index, 0]

    x_list.append(data)
    y_list.append([truth])


def convert_data(x_list, y_list):

    """ This function once did more; will keep for now.
    """

    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):

    """ Not quite sure what's going on here, will investigate.
    """

    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y
