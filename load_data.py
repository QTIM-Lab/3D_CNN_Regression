from __future__ import division

import os
import glob

import numpy as np
import tables
import nibabel as nib

from image_utils import nifti_2_numpy

def fetch_data_files(data_dir, input_modalities, input_groundtruth):

    """ This function gathers all modalities plus groundtruth for a given patient into
        a tuple, and then creates a list of tuples representing a list of all patients. Currently, groundtruth is just coded as "the last" file in a tuple for each case. In the future, maybe make two tuples for inputs and outputs.

        TODO: Smart error-catching.
    """

    data_files = []
    truth_files = []
    print os.path.join(data_dir, "*")

    for subject_dir in glob.glob(os.path.join(data_dir, "*")):

        # Input Modalities
        subject_files = []
        for modality in input_modalities:
            target_file = glob.glob(os.path.join(subject_dir, '*' + modality + '*'))
            try:
                subject_files.append(target_file[0])
            except:
                break
        if len(subject_files) == len(input_modalities):
            data_files.append(tuple(subject_files))

        # Groundtruth "Modalities"
        subject_files = []
        for modality in input_groundtruth:
            target_file = glob.glob(os.path.join(subject_dir, '*' + modality + '*'))
            try:
                subject_files.append(target_file[0])
            except:
                break
        if len(subject_files) == len(input_groundtruth):
            truth_files.append(tuple(subject_files))

    return data_files, truth_files

def write_data_to_file(input_data_files, input_groundtruth_files, output_hdf5_filepath, image_shape, patches=False, patch_num=None, patch_shape=None, patch_extraction_conditions=[]):

    # This is a bit hacky, with the patches. Think about data structure.

    # TODO: Modifiy this for multiple outputs, e.g.
    # in the DCE-MRI parameters case.
    if patches:
        row_num = patch_num
        row_shape = patch_shape
    else:
        row_num = len(input_data_files)
        row_shape = image_shape

    input_modality_num, groundtruth_modality_num = len(input_data_files[0]), len(input_groundtruth_files[0])

    try:
        hdf5_file, data_storage, truth_storage = create_image_hdf5_file(output_hdf5_filepath, input_modality_num=input_modality_num, groundtruth_modality_num=groundtruth_modality_num, row_num=row_num, row_shape=row_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(output_hdf5_filepath)
        raise e

    if patches:
        write_image_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, image_shape, input_modality_num=input_modality_num)
    else:
        write_patch_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, patch_shape=patch_shape, patch_num=patch_num, input_modality_num=input_modality_num, patch_extraction_conditions=patch_extraction_conditions)

    hdf5_file.close()
    return output_hdf5_filepath

def create_image_hdf5_file(output_filepath, input_modality_num, groundtruth_modality_num, row_num, row_shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    # Input data has multiple 'channels' i.e. modalities.
    data_shape = tuple([0, input_modality_num] + list(row_shape))

    # Ground truth data only has one channel, ground truth modality.
    truth_shape = tuple([0, groundtruth_modality_num] + list(row_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=row_num)

    # For the classification case, dtype is UInt8Atom. For regression, we do Float again.
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape, filters=filters, expectedrows=row_num)

    return hdf5_file, data_storage, truth_storage

def write_image_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, image_shape, input_modality_num):

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

def extract_patch(input_image_stack, patch_shape, patch_extraction_condition=None):

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