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

    for subject_dir in glob.glob(os.path.join(data_dir, "*")):
        subject_files = []
        for modality in input_modalities + input_groundtruth:
            target_file = glob.glob(os.path.join(subject_dir, '*' + modality + '*'))
            try:
                subject_files.append(target_file[0])
            except:
                break
        if len(subject_files) == len(input_modalities) + len(input_groundtruth):
            data_files.append(tuple(subject_files))

    return data_files

def write_data_to_file(input_data_files, output_hdf5_filepath, image_shape):

    # TODO: Modifiy this for multiple outputs, e.g.
    # in the DCE-MRI parameters case.
    n_cases = len(input_data_files)
    n_modalities = len(input_data_files[0]) - 1

    try:
        hdf5_file, data_storage, truth_storage = create_image_hdf5_file(output_hdf5_filepath, modality_num=n_modalities, case_num=n_cases, input_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(output_hdf5_filepath)
        raise e

    write_image_data_to_hdf5(input_data_files, data_storage, truth_storage, image_shape, n_modalities=n_modalities)
    
    hdf5_file.close()
    return output_hdf5_filepath

def create_image_hdf5_file(output_filepath, modality_num, case_num, input_shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    # Input data has multiple 'channels' i.e. modalities.
    data_shape = tuple([0, modality_num] + list(input_shape))

    # Ground truth data only has one channel, ground truth modality.
    truth_shape = tuple([0, 1] + list(input_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=case_num)

    # For the classification case, dtype is UInt8Atom. For regression, we do Float again.
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape, filters=filters, expectedrows=case_num)

    return hdf5_file, data_storage, truth_storage

def write_image_data_to_hdf5(image_files, data_storage, truth_storage, image_shape, n_modalities):

    for set_of_files in image_files:

        subject_data = read_image_files(set_of_files)

        # I don't intuitively understand what's going on with new axis here, but the dimensions seem to work out.
        data_storage.append(subject_data[:n_modalities][np.newaxis])
        truth_storage.append(np.asarray(subject_data[n_modalities][np.newaxis][np.newaxis]))

    return data_storage, truth_storage

def read_image_files(image_files):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    return np.stack([image for image in image_list])
