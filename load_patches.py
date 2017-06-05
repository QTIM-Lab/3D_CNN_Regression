from __future__ import division

import os
import glob

import numpy as np
import tables
import nibabel as nib

from image_utils import nifti_2_numpy

def write_patches_to_file(input_data_files, output_hdf5_filepath, image_shape, patch_num, patch_shape):

    # TODO: Modifiy this for multiple outputs, e.g.
    # in the DCE-MRI parameters case.
    num_cases = len(input_data_files)
    modality_num = len(input_data_files[0]) - 1

    try:
        hdf5_file, data_storage, truth_storage = create_patches_hdf5_file(output_hdf5_filepath, modality_num=modality_num, patch_num=patch_num, patch_shape=patch_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(output_hdf5_filepath)
        raise e

    write_patch_data_to_hdf5(input_data_files, data_storage, truth_storage, patch_shape=patch_shape, patch_num=patch_num, modality_num=modality_num)
    
    hdf5_file.close()

    return output_hdf5_filepath

def create_patches_hdf5_file(output_filepath, modality_num, patch_num, patch_shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    # Input data has multiple 'channels' i.e. modalities.
    data_shape = tuple([0, modality_num] + list(patch_shape))

    # Ground truth data only has one channel, ground truth modality.
    truth_shape = tuple([0, 1] + list(patch_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=patch_num)

    # For the classification case, dtype is UInt8Atom. For regression, we do Float again.
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape, filters=filters, expectedrows=patch_num)

    return hdf5_file, data_storage, truth_storage

def write_patch_data_to_hdf5(image_files, data_storage, truth_storage, patch_shape, patch_num, modality_num):

    total_patches = 0
    current_image_index = 0
    patches_per_image = int(np.ceil(patch_num/len(image_files)))
    print len(image_files), patches_per_image, patch_num
    sample_images = read_image_files(image_files[0])

    while total_patches < patch_num:

        patch_data = extract_patch(sample_images, patch_shape)

        data_storage.append(patch_data[:modality_num][np.newaxis])
        truth_storage.append(np.asarray(patch_data[modality_num][np.newaxis][np.newaxis]))

        if (total_patches + 1) % patches_per_image == 0:
            current_image_index += 1
            if not current_image_index >= len(image_files):
                sample_images = read_image_files(image_files[current_image_index])

        total_patches += 1

    return data_storage, truth_storage

def extract_patch(input_image_stack, patch_shape, background_max_ratio=.5, mask_value=0):

    """ The start of what will likely need to be a
        much more complex program to extract patches.

        TODO: Make patches be chosen not just by the top
        left corner.
    """

    background_patch = True
    # patch_vox_count = float(np.product((input_image_stack.shape[0],) + patch_shape))
    patch_vox_count = float(np.product(patch_shape))

    while background_patch:

        corner = [np.random.randint(0, max_dim) for max_dim in input_image_stack.shape[1:]]
        patch_slice = [slice(None)] + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]
        patch = input_image_stack[patch_slice]

        # print 'corner', corner
        pad_dims = [(0,0)]
        # print 'patch shape', patch.shape
        for idx, dim in enumerate(patch.shape[1:]):
            pad_dims += [(0, patch_shape[idx]-dim)]

        # print 'padding dimensions', pad_dims
        patch = np.lib.pad(patch, tuple(pad_dims), 'edge')

        # if float((patch == 0).sum()) / patch_vox_count < background_max_ratio:
            # background_patch = False

        if float((patch[-1,...] <= .4).sum()) / patch_vox_count < background_max_ratio:
            background_patch = False    

    return patch

def read_image_files(image_files):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    return np.stack([image for image in image_list])
