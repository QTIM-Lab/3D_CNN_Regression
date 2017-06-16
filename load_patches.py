from __future__ import division

import os
import glob

import numpy as np
import tables
import nibabel as nib

from image_utils import nifti_2_numpy

def write_patches_to_file(input_data_files, input_groundtruth_files, output_hdf5_filepath, image_shape, patch_num, patch_shape, patch_extraction_conditions=[]):

    # TODO: Modifiy this for multiple outputs, e.g.
    # in the DCE-MRI parameters case.
    num_cases = len(input_data_files)
    input_modality_num, groundtruth_modality_num = len(input_data_files[0]), len(input_groundtruth_files[1])

    try:
        hdf5_file, data_storage, truth_storage = create_patches_hdf5_file(output_hdf5_filepath, input_modality_num=input_modality_num, groundtruth_modality_num=groundtruth_modality_num, patch_num=patch_num, patch_shape=patch_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(output_hdf5_filepath)
        raise e

    write_patch_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, patch_shape=patch_shape, patch_num=patch_num, input_modality_num=input_modality_num, patch_extraction_conditions=patch_extraction_conditions)
    
    hdf5_file.close()

    return output_hdf5_filepath

def create_patches_hdf5_file(output_filepath, input_modality_num, groundtruth_modality_num, patch_num, patch_shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    # Input data has multiple 'channels' i.e. modalities.
    data_shape = tuple([0, input_modality_num] + list(patch_shape))

    # Ground truth data only has one channel, ground truth modality.
    truth_shape = tuple([0, groundtruth_modality_num] + list(patch_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=patch_num)

    # For the classification case, dtype is UInt8Atom. For regression, we do Float again.
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape, filters=filters, expectedrows=patch_num)

    return hdf5_file, data_storage, truth_storage

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
            print patch_data.shape
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

        TODO: Make patches be chosen not just by the top
        left corner.
    """

    acceptable_patch = False

    while not acceptable_patch:

        corner = [np.random.randint(0, max_dim) for max_dim in input_image_stack.shape[1:]]
        patch_slice = [slice(None)] + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]
        patch = input_image_stack[patch_slice]


        pad_dims = [(0,0)]
        # print 'patch shape', patch.shape
        for idx, dim in enumerate(patch.shape[1:]):
            pad_dims += [(0, patch_shape[idx]-dim)]

        # print 'padding dimensions', pad_dims
        patch = np.lib.pad(patch, tuple(pad_dims), 'edge')

        # print patch_extraction_condition

        if patch_extraction_condition:
            acceptable_patch = patch_extraction_condition[0](patch)
        else:
            acceptable_patch = True

        # print acceptable_patch

    return patch

def read_image_files(image_files):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    return np.stack([image for image in image_list])
