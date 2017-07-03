import os
import pickle
from random import shuffle

import numpy as np

from data_utils import pickle_dump, pickle_load

def get_data_generator(data_file, batch_size, data_labels):

    if isinstance(data_labels, basestring):
        data_labels = [data_labels]

    num_steps = getattr(data_file.root, data_labels[0]).shape[0]
    output_data_generator = data_generator(data_file, range(num_steps), data_labels=data_labels, batch_size=batch_size)

    return output_data_generator, num_steps // batch_size

def data_generator(data_file, index_list, data_labels, batch_size=1):

    """ TODO: Investigate how generators even work?! And yield.
    """

    while True:
        data_lists = [[] for i in data_labels]
        shuffle(index_list)

        for index in index_list:

            add_data(data_lists, data_file, index, data_labels)

            # print 'STATUS DATA LISTS'

            if len(data_lists[0]) == batch_size:

                yield tuple([np.asarray(data_list) for data_list in data_lists])
                data_lists = [[] for i in data_labels]

def add_data(data_lists, data_file, index, data_labels):

    for data_idx, data_label in enumerate(data_labels):
        data = getattr(data_file.root, data_label)[index]
        data_lists[data_idx].append(data)