
import os

import nibabel as nib
import numpy as np
import tables

# from .training import load_old_model
# from .utils import pickle_load

from model import load_old_model
from image_utils import save_numpy_2_nifti

def model_predict_patches(data_file, input_data, patch_shape, repetitions=16, test_batch_size=100, output_data=None, output_shape=None, model=None, model_file=None, output_directory=None):

    """ TODO: Make work for multiple inputs and outputs.
        TODO: Interact with data group interface
        TODO: Pass output filenames to hdf5 files.
    """

    print 'OUTPUT_SHAPE', output_shape, '\n'

    # Create output directory. If not provided, output into original patient folder.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Load model.
    if model is None and model_file is None:
        print 'Error. Please provide either a model object or a model filepath.'
    elif model is None:
        model = load_old_model(model_file)

    # TODO: Add check in case an object is passed in.
    # input_data_object = self.data_groups[input_data_group]

    getattr(data_file.root, input_data).shape[0]

    # Iterate through cases and predict.
    for case_idx, case_name in enumerate(xrange(getattr(data_file.root, input_data).shape[0])):

        print 'Working on image.. ', case_idx, 'in', case_name

        test_input = np.asarray([getattr(data_file.root, input_data)[case_idx]])

        if output_data is not None:
            test_output = np.asarray([getattr(data_file.root, output_data)[case_idx]])
            save_numpy_2_nifti(np.squeeze(test_output), output_filepath=os.path.join(output_directory, 'TESTCASE_' + str(case_idx).zfill(3) + '_TRUTH.nii.gz'))

        for modality_num in xrange(test_input.shape[1]):
            save_numpy_2_nifti(np.squeeze(test_input[:,modality_num,...]), output_filepath=os.path.join(output_directory, 'TESTCASE_' + str(case_idx).zfill(3) + '_MODALITY_' + str(modality_num) + '.nii.gz'))

        if output_data is None and output_shape is None:
            print 'Currently, you must provide either a reference ground truth data or a reference output shape to perform a prediction.'
        elif output_data is None:
            final_image = np.zeros(output_shape)
        else:
            final_image = np.zeros_like(test_output)

        for rep_idx in xrange(repetitions):

            print 'PATCH GRID REPETITION # ..', rep_idx

            offset_slice = [slice(None)]*2 + [slice(rep_idx, None, 1)] * (test_input.ndim - 2)
            repatched_image = np.zeros_like(final_image[offset_slice])
            corners_list = patchify_image(test_input[offset_slice], [test_input[offset_slice].shape[1]] + list(patch_shape))

            for corner_list_idx in xrange(0, len(corners_list), test_batch_size):

                corner_batch = corners_list[corner_list_idx:corner_list_idx+test_batch_size]
                input_patches = grab_patch(test_input[offset_slice], corners_list[corner_list_idx:corner_list_idx+test_batch_size], patch_shape)
                prediction = model.predict(input_patches)

                for corner_idx, corner in enumerate(corner_batch):
                    insert_patch(repatched_image, prediction[corner_idx, ...], corner)

            if rep_idx == 0:
                final_image = np.copy(repatched_image)
            else:
                final_image[offset_slice] = final_image[offset_slice] + (1.0 / (rep_idx)) * (repatched_image - final_image[offset_slice])
        
        final_image = np.around(np.squeeze(final_image))

        print 'Sum of output...', np.sum(final_image[0,...]), np.sum(final_image[1,...]), np.sum(final_image[2,...])

        # Multi-label images. TODO: Standardize this.
        composite_final_image = final_image[0,...]
        composite_final_image[final_image[1,...] > 0] = 2
        composite_final_image[final_image[2,...] > 0] = 3

        save_numpy_2_nifti(composite_final_image, output_filepath=os.path.join(output_directory, 'TESTCASE_' + str(case_idx).zfill(3) + '_PREDICT.nii.gz'))

    data_file.close() 

def run_validation_case(output_dir, model_file, data_file):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_old_model(model_file)

    for case_idx in xrange(data_file.root.data.shape[0]):
        test_input = np.asarray([data_file.root.data[case_idx]])
        test_truth = np.asarray([data_file.root.truth[case_idx]])

        save_numpy_2_nifti(np.squeeze(test_truth), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_idx).zfill(3) + '_TRUTH.nii.gz'))
        prediction = model.predict(test_input)
        save_numpy_2_nifti(np.squeeze(prediction), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_idx).zfill(3) + '_PREDICT.nii.gz'))

    data_file.close()

def patchify_image(input_data, patch_shape, offset=(0,0,0,0), batch_dim=True, return_patches=False):

    """ VERY wonky. Patchs an image of arbitrary dimension, but
        has some interesting assumptions built-in about batch sizes,
        channels, etc.

        TODO: Make this function able to iterate forward or backward.
    """

    corner = [0] * len(input_data.shape[1:])

    if return_patches:
        patch = grab_patch(input_data, corner, patch_shape)
        patch_list = [[corner[:], patch[:]]]
    else:
        patch_list = [corner[:]]

    finished = False

    while not finished:

        if return_patches:
            patch = grab_patch(input_data, corner, patch_shape)
            patch_list += [[corner[:], patch[:]]]
        else:
            patch_list += [corner[:]]

        for idx, corner_dim in enumerate(corner):

            # Advance corner stride
            if idx == 0:
                corner[idx] += patch_shape[idx]

            # Finish patchification
            if idx == len(corner) - 1 and corner[idx] == input_data.shape[-1]:
                finished = True
                continue

            # Push down a dimension.
            if corner[idx] == input_data.shape[idx+1]:
                corner[idx] = 0
                corner[idx+1] += patch_shape[idx+1]

            # Reset patch at edge.
            elif corner[idx] > input_data.shape[idx+1] - patch_shape[idx]:
                corner[idx] = input_data.shape[idx+1] - patch_shape[idx]

    return patch_list

def grab_patch(input_data, corner_list, patch_shape):

    """ Given a corner coordinate, a patch_shape, and some input_data, returns a patch or array of patches.
    """

    output_patches = np.zeros(((len(corner_list),input_data.shape[1]) + patch_shape))

    for corner_idx, corner in enumerate(corner_list):
        output_slice = [slice(None)]*2 + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner[1:])]
        output_patches[corner_idx, ...] = input_data[output_slice]

    return output_patches


def insert_patch(input_data, patch, corner):

    patch_shape = patch.shape[1:]

    patch_slice = [slice(None)]*2 + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner[1:])]
    
    input_data[patch_slice] = patch

    return

def calculate_prediction_msq(output_dir):

    """ Calculate mean-squared error for the predictions folder.
    """

if __name__ == '__main__':
    pass