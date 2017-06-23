from __future__ import division

import os
import glob

import numpy as np
import tables
import nibabel as nib

from image_utils import nifti_2_numpy

class DataGroups(object):

    def __init__(self, data_directory, modality_dict, spreadsheet_dict=None, value_dict=None, case_list=None):

        # Input vars
        self.data_directory = os.path.abspath(data_directory)
        self.modality_dict = modality_dict
        self.spreadsheet_dict = spreadsheet_dict
        self.value_dict = value_dict
        self.case_list = None

        # Empty vars
        self.data = {}

        # Data-Checking, e.g. duplicate data keys.

    def fill_data_groups(self):

        print 'Gathering image data from...', self.data_directory

        #####
        # TODO: Add section for spreadsheets.
        # TODO: Add section for values.
        #####

        # Imaging modalities added.
        for subject_dir in sorted(glob.glob(os.path.join(self.data_directory, "*/"))):

            if self.case_list is not None:
                if os.path.basename(subject_dir) in self.case_list:
                    continue

            for modality_group in self.modality_dict:

                self.data[modality_group] = []
                modality_group_files = []

                for modality in self.modality_dict[modality_group]:
                    target_file = glob.glob(os.path.join(subject_dir, modality))
                    try:
                        modality_group_files.append(target_file[0])
                    except:
                        break

                if len(modality_group_files) == len(self.modality_dict[modality_group]):
                    self.data[modality_group].append(tuple(modality_group_files))

    def write_data_to_file(self, output_filetype='hdf5'):
        return

    def augment_data(self, augmentation=Augmentation(), data_groups=[]):
        return

class Augmentation(object):

    def __init__(self):
        return

class ExtractPatches(Augmentation):

    def augment(input_image_stack, patch_shape, patch_extraction_condition=None):

        """ The start of what will likely need to be a
            much more complex program to extract patches.

            Parameters
            ----------
            input_image_stack: ndarray
                Array of format [nmodalities, (image_shape)] from which to extract a patch.
            patch_shape: ndarray
                Patch_shape the same shape as (image_shape) to extract.
            patch_extraction_condition: function
                A function that takes in a patch and returns "True" or "False". If this condition
                is not met, a new patch is extracted.

            TODO: Make patches be chosen not just by the top
            left corner.
        """

        acceptable_patch = False

        while not acceptable_patch:

            corner = [np.random.randint(0, max_dim) for max_dim in input_image_stack.shape[1:]]
            patch_slice = [slice(None)] + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]
            patch = input_image_stack[patch_slice]

            pad_dims = [(0,0)]
            for idx, dim in enumerate(patch.shape[1:]):
                pad_dims += [(0, patch_shape[idx]-dim)]

            patch = np.lib.pad(patch, tuple(pad_dims), 'edge')

            if patch_extraction_condition:
                acceptable_patch = patch_extraction_condition[0](patch)
            else:
                acceptable_patch = True

        return patch


def read_image_files(image_files):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    return np.stack([image for image in image_list])




def fetch_data_files(data_dir, modality_dict, spreadsheet_dict=None, value_dict=None):

    """ This function gathers all modalities plus groundtruth for a given patient into
        a tuple, and then creates a list of tuples representing a list of all patients. Currently, groundtruth is just coded as "the last" file in a tuple for each case. In the future, maybe make two tuples for inputs and outputs.

        TODO: Smart error-catching.
    """

    print os.path.join(data_dir, "*")

    data_groups = {}

    # Get file modalities
    for subject_dir in glob.glob(os.path.join(data_dir, "*/")):

        for modality_group in modality_dict:

            data_groups[modality_group] = []
            modality_group_files = []

            for modality in modality_dict[modality_group]:
                target_file = glob.glob(os.path.join(subject_dir, modality))
                try:
                    modality_group_files.append(target_file[0])
                except:
                    break
            if len(modality_group_files) == len(modality_dict[modality_group]):
                data_groups[modality_group].append(tuple(modality_group_files))

    # TODO: Add section for spreadsheets.
    # TODO: Add section for values.
    #####

    return data_groups

def write_data_to_file(data_groups, output_hdf5_filepath, augmentation_dicts=None):

    """ Writes a list of filenames to an hdf5 file, subject to data augmentations. Data
        files can acquired using fetch_data_files.
    """

    example_augmentation_dict = [{'augmentation_functions': test_func,
                                    'params': [1,2],
                                    'augmentation_multiplier': 1,
                                    'augmentation_total': None,
                                    'augmentation_output_shapes': None}]

    # Get number of cases from an arbitrary data group.
    num_cases = len(data_groups[data_groups.keys()[0]])
    # Get modality numbers from all groups. This will currently fail for spreadsheet/value groups.
    modality_num = {}
    for key in data_groups:
        modality_nums[key] = len(data_groups[key][0])

    # Get output size for list of augmentations.
    for augmentation in augmentation_dict:

        # Get output rows
        try:
            multiplier = augmentation['augmentation_multiplier']
        except:
            multiplier = 1
        # Get data amount / cap
        try:
            total = augmentation['augmentation_total']
        except:
            total = None
        # If multiplier goes over "total", use total
        if total is None:
            num_cases *= multiplier
        elif ((num_cases * multiplier) - num_cases) > total:
            num_cases += total
        else:
            num_cases *= multiplier

        # Get output shape
        try:
            input_shape = augmentation['augmentation_input_shape']
        except:
            pass

    # Create Data File
    try:
        hdf5_file, data_storage, truth_storage = create_image_hdf5_file(output_hdf5_filepath, input_modality_num=input_modality_num, groundtruth_modality_num=groundtruth_modality_num, row_num=num_cases, input_shape=input_shape, groundtruth_shape=groundtruth_shape)
    except Exception as e:
        os.remove(output_hdf5_filepath)
        raise e

    # Write data
    write_image_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, image_shape, groundtruth_shape, augmentation_dicts)

    hdf5_file.close()
    return output_hdf5_filepath

def create_image_hdf5_file(output_filepath, input_modality_num, groundtruth_modality_num, row_num, input_shape, groundtruth_shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    # Input data has multiple 'channels' i.e. modalities.
    data_shape = tuple([0, input_modality_num] + list(input_shape))

    # Ground truth data only has one channel, ground truth modality.
    truth_shape = tuple([0, groundtruth_modality_num] + list(groundtruth_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=row_num)

    # For the classification case, dtype is UInt8Atom. For regression, we do Float again.
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape, filters=filters, expectedrows=row_num)

    return hdf5_file, data_storage, truth_storage

def write_image_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, image_shape, groundtruth_shape, augmentation_dicts):

    for image_idx in xrange(len(input_data_files)):

        subject_data = read_image_files(input_data_files[image_idx] + input_groundtruth_files[image_idx])

        # I don't intuitively understand what's going on with new axis here, but the dimensions seem to work out.
        data_storage.append(subject_data[:input_modality_num][np.newaxis])
        truth_storage.append(np.asarray(subject_data[input_modality_num:][np.newaxis]))

    return data_storage, truth_storage

def write_patch_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, patch_shape, patch_num, input_modality_num, patch_extraction_conditions=None):
 
    """ This program should be totally rewritten..
    """

    for image_idx in xrange(len(input_data_files)):

        # Get number of patches to analyze.
        # Note that this short-shrifts the last image for patches, because I'm exhausted by integer division right now.
        if image_idx == len(input_data_files) - 1:
            patches_per_image = int(patch_num - (len(input_data_files) - 1) * np.ceil(patch_num/len(input_data_files)))
        else:
            patches_per_image = int(np.ceil(patch_num/len(input_data_files)))

        # Load data into memory
        image_data = read_image_files(input_data_files[image_idx] + input_groundtruth_files[image_idx])

        # Patch extraction conditions
        start_idx = 0
        condition_list = [-1] * patches_per_image
        for condition_idx, patch_extraction_condition in enumerate(patch_extraction_conditions):
            end_idx = start_idx + int(np.ceil(patch_extraction_condition[1]*patches_per_image))
            condition_list[start_idx:end_idx] = [condition_idx]*(end_idx-start_idx)
            start_idx = end_idx

        print 'PATCH TYPES: ', condition_list

        for condition in condition_list:

            # Extract patch..
            if condition < 0:
                patch_data = extract_patch(image_data, patch_shape)
            else:
                patch_data = extract_patch(image_data, patch_shape, patch_extraction_conditions[condition])
            
            # Save data in batch format..
            data_storage.append(patch_data[:input_modality_num][np.newaxis])
            truth_storage.append(np.asarray(patch_data[input_modality_num:][np.newaxis]))

    return data_storage, truth_storage

def test_func():
    return