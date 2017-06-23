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
        self.augmentations = augmentations

        # Empty vars
        self.data_groups = {}
        self.data_shape = None
        self.data_shape_augment = None

        # Data-Checking, e.g. duplicate data keys, data_shape

    def verify_data_shape(self):
        return

    def append_augmentation(self, augmentation):
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

            self.data_groups[modality_group] = DataGroup(modality_group)

            for subject_dir in sorted(glob.glob(os.path.join(self.data_directory, "*/"))):

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
                    print tuple(modality_group_files)
                    self.data_groups[modality_group].add_case(tuple(modality_group_files), os.path.abspath(subject_dir))

    def write_data_to_file(self, output_filepath=None, data_groups=None):

        if data_groups is None:
            data_groups = self.data_groups.keys()

        if output_filepath is None:
            output_filepath = os.path.join(self.data_directory, '_'.join(modalities) + '.hdf5')

        # Create Data File
        try:
            hdf5_file = create_hdf5_file(output_filepath, data_groups, self)
        except Exception as e:
            os.remove(output_filepath)
            raise e

        # Write data
        write_image_data_to_hdf5(data_groups, data_storages, self)

        hdf5_file.close()
        return output_hdf5_filepath

class DataGroup(object):

    def __init__(self, label):

        self.label = label
        self.augmentations = []
        self.data = []
        self.case_names = []

        self.storage = None

        self.num_cases = 0

    def add_case(self, item, case_name):
        self.data.append(item)
        self.case_names.append(case_name)
        self.num_cases = len(self.data)

    def add_augmentations(self, augmentation):
        self.augmentations.append(Augmentation)

    def get_augment_num_shape(self):

        output_num = len(self.data)
        output_shape = get_shape(self)

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
            return nifti_2_numpy(self.data[0]).shape

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
        num_cases, input_shape = data_group.get_augment_num_shape()
        modalities = data_group.get_modalities()

        # Input data has multiple 'channels' i.e. modalities.
        data_shape = tuple([0, modalities] + list(input_shape))
        data_storages += [hdf5_file.create_earray(hdf5_file.root, data_group.label, tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)]
        data_group.storage = data_storages[-1]

    return hdf5_file, data_storages

def write_image_data_to_hdf5(data_groups, data_collection):

    for data_group_label in data_groups:

        data_group = data_collection.data_groups[data_group_label]

        for image_idx in xrange(len(data_group.data)):

            subject_data = read_image_files(data_group.data)[:][np.newaxis]

            # This list-y-ness of this part is questionable, i.e. when its defined as a list.
            subject_data = data_group.augment(subject_data)

            # I don't intuitively understand what's going on with new axis here, but the dimensions seem to work out.
            for subject_data_i in subject_data:
                data_group.data_storage.append(subject_data_i)

def read_image_files(image_files):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    return np.stack([image for image in image_list])

if __name__ == '__main__':
    pass