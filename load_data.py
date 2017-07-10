from __future__ import division

import os
import glob

import numpy as np
import tables
import nibabel as nib

from image_utils import nifti_2_numpy
from augment import *

class DataCollection(object):

    def __init__(self, data_directory, modality_dict, spreadsheet_dict=None, value_dict=None, case_list=None, augmentations=None):

        # Input vars
        self.data_directory = os.path.abspath(data_directory)
        self.modality_dict = modality_dict
        self.spreadsheet_dict = spreadsheet_dict
        self.value_dict = value_dict
        self.case_list = case_list

        # Special behavior for augmentations
        self.augmentations = []
        self.cases = []

        # Augmentations

        # Empty vars
        self.data_groups = {}
        self.data_shape = None
        self.data_shape_augment = None

        # Data-Checking, e.g. duplicate data keys, data_shape

    def verify_data_shape(self):
        return

    def append_augmentation(self, augmentation_group, data_groups=None):

        for data_group_label in augmentation_group.augmentation_dict.keys():
            self.data_groups[data_group_label].append_augmentation(augmentation_group.augmentation_dict[data_group_label])

            augmentation_group.augmentation_dict[data_group_label].append_data_group(self.data_groups[data_group_label])
            augmentation_group.augmentation_dict[data_group_label].initialize_augmentation()

        # Don't like this list method, find a more explicity way.
        self.augmentations.append(augmentation_group)

        return

    def fill_data_groups(self):

        print 'Gathering image data from...', self.data_directory

        #####
        # TODO: Add section for spreadsheets.
        # TODO: Add section for values.
        #####

        # Imaging modalities added.

        for modality_group in self.modality_dict:

            # print modality_group
            if modality_group not in self.data_groups.keys():
                self.data_groups[modality_group] = DataGroup(modality_group)

        for subject_dir in sorted(glob.glob(os.path.join(self.data_directory, "*/"))):

            self.cases.append(os.path.abspath(subject_dir))

            for modality_group in self.modality_dict:

                if self.case_list is not None:
                    if os.path.basename(subject_dir) not in self.case_list:
                        continue

                modality_group_files = []

                for modality in self.modality_dict[modality_group]:
                    target_file = glob.glob(os.path.join(subject_dir, modality))
                    try:
                        modality_group_files.append(target_file[0])
                    except:
                        break

                if len(modality_group_files) == len(self.modality_dict[modality_group]):
                    self.data_groups[modality_group].add_case(tuple(modality_group_files), os.path.abspath(subject_dir))

    def write_data_to_file(self, output_filepath=None, data_groups=None):

        """ Interesting question: Should all passed data_groups be assumed to have equal size? Nothing about hdf5 requires that, but it makes things a lot easier to assume.
        """

        if data_groups is None:
            data_groups = self.data_groups.keys()

        if output_filepath is None:
            output_filepath = os.path.join(self.data_directory, 'data.hdf5')

        # Create Data File
        # try:
        hdf5_file = create_hdf5_file(output_filepath, data_groups, self)
        # except Exception as e:
            # os.remove(output_filepath)
            # raise e

        # Write data
        self.write_image_data_to_hdf5(data_groups)

        hdf5_file.close()

    def write_image_data_to_hdf5(self, data_groups):

        """ Some of the syntax around data groups can be cleaned up in this function.
        """

        data_group_objects = [self.data_groups[label] for label in data_groups]

        # Will cases always be ordered..?
        for case_idx, case_name in enumerate(self.cases):

            print 'Working on image.. ', case_idx, 'in', case_name

            all_cases = {}

            # Check for missing data.
            missing_case = False
            for data_group in data_group_objects:

                try:
                    case_index = data_group.cases.index(case_name)
                except:
                    missing_case = True
                    missing_data_group = data_group.label
                    break
                data_group.base_case = read_image_files(data_group.data[case_index])[:][np.newaxis]
                data_group.current_case = data_group.base_case

            if missing_case:
                print 'Missing case', case_name, 'in data group', missing_data_group, '. Skipping this case..'
                continue

            print '\n'

            self.recursive_augmentation(data_group_objects)


    def recursive_augmentation(self, data_groups, augmentation_num=0):

        """ This function baldly reveals my newness at recursion..
        """

        print 'BEGIN RECURSION FOR AUGMENTATION NUM', augmentation_num

        if augmentation_num == len(self.augmentations):
            for data_group in data_groups:
                data_group.write_to_storage()
            return

        else:

            for iteration in xrange(self.augmentations[augmentation_num].total_iterations):

                print 'AUGMENTATION NUM', augmentation_num, 'ITERATION', iteration

                for data_group in data_groups:

                    if augmentation_num == 0:
                        data_group.augmentation_cases[augmentation_num] = self.augmentations[augmentation_num].augmentation_dict[data_group.label].augment(data_group.base_case)
                    else:
                        data_group.augmentation_cases[augmentation_num] = self.augmentations[augmentation_num].augmentation_dict[data_group.label].augment(data_group.augmentation_cases[augmentation_num-1])

                    data_group.current_case = data_group.augmentation_cases[augmentation_num]
                    data_group.augmentation_num += 1

                self.recursive_augmentation(data_groups, augmentation_num+1)

                print 'FINISH RECURSION FOR AUGMENTATION NUM', augmentation_num+1

                for data_group in data_groups:
                    if augmentation_num == 0:
                        data_group.current_case = data_group.base_case
                    else:
                        data_group.current_case = data_group.augmentation_cases[augmentation_num - 1]

                for data_group in data_groups:        
                    self.augmentations[augmentation_num].augmentation_dict[data_group.label].iterate()

        for data_group in data_groups:
            data_group.augmentation_num -= 1

        return

class DataGroup(object):

    def __init__(self, label):

        self.label = label
        self.augmentations = []
        self.data = []
        self.cases = []

        self.base_case = None
        self.augmentation_cases = []
        self.current_case = None

        self.augmentation_num = -1

        self.data_storage = None

        self.num_cases = 0

    def add_case(self, item, case_name):
        self.data.append(item)
        self.cases.append(case_name)
        self.num_cases = len(self.data)

    def append_augmentation(self, augmentation):
        self.augmentations.append(augmentation)
        self.augmentation_cases.append([])

    def get_augment_num_shape(self):

        output_num = len(self.data)
        output_shape = self.get_shape()

        # Get output size for list of augmentations.
        for augmentation in self.augmentations:

            # Error Catching
            if augmentation.total is None and augmentation.multiplier is None:
                continue

            # If multiplier goes over "total", use total
            if augmentation.total is None:
                output_num *= augmentation.multiplier
            elif ((num_cases * augmentation.multiplier) - num_cases) > augmentation.total:
                output_num += augmentation.total
            else:
                output_num *= augmentation.multiplier

            # Get output shape, if it changes
            if augmentation.output_shape is not None:
                output_shape = augmentation.output_shape

        return output_num, output_shape

    def get_shape(self):

        # TODO: Add support for non-nifti files.
        # Also this is not good. Perhaps specify shape in input?

        if self.data == []:
            return (0,)
        else:
            return nifti_2_numpy(self.data[0][0]).shape

    def get_modalities(self):
        if self.data == []:
            return 0
        else:
            return len(self.data[0])

    def augment(self, input_data):

        output_data = [input_data]

        for augmentatation in self.augmentations:

            output_data = augmentation.augment(input_data)

        return output_data

    def write_to_storage(self):
        self.data_storage.append(self.current_case)

def create_hdf5_file(output_filepath, data_groups, data_collection):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    data_storages = []

    for data_group_label in data_groups:

        data_group = data_collection.data_groups[data_group_label]

        num_cases, output_shape = data_group.get_augment_num_shape()
        modalities = data_group.get_modalities()

        # Input data has multiple 'channels' i.e. modalities.
        data_shape = tuple([0, modalities] + list(output_shape))
        print data_group.label, data_shape
        data_group.data_storage = hdf5_file.create_earray(hdf5_file.root, data_group.label, tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    return hdf5_file

def read_image_files(image_files):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    return np.stack([image for image in image_list])

if __name__ == '__main__':
    pass