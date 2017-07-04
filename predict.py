
import os

import nibabel as nib
import numpy as np
import tables

# from .training import load_old_model
# from .utils import pickle_load

from model import load_old_model
from image_utils import save_numpy_2_nifti

def get_test_indices(testing_file):
    return pickle_load(testing_file)

def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])

def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)

def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)

def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)

def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine)

def run_validation_case(output_dir, model_file, data_file):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_old_model(model_file)

    for case_num in xrange(data_file.root.data.shape[0]):
        test_data = np.asarray([data_file.root.data[case_num]])
        test_truth = np.asarray([data_file.root.truth[case_num]])

        save_numpy_2_nifti(np.squeeze(test_truth), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_TRUTH.nii.gz'))
        prediction = model.predict(test_data)
        save_numpy_2_nifti(np.squeeze(prediction), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_PREDICT.nii.gz'))

    data_file.close()

def run_validation_case_patches(output_dir, model_object, model_file, data_file, patch_shape, repetitions=16, test_batch=75):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # model = load_old_model(model_file)
    model = model_object

    for case_num in xrange(data_file.root.input_modalities.shape[0]):

        test_data = np.asarray([data_file.root.input_modalities[case_num]])
        test_truth = np.asarray([data_file.root.ground_truth[case_num]])

        save_numpy_2_nifti(np.squeeze(test_truth), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_TRUTH.nii.gz'))
        for modality_num in xrange(test_data.shape[1]):
            save_numpy_2_nifti(np.squeeze(test_data[:,modality_num,...]), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_MODALITY_' + str(modality_num) + '.nii.gz'))

        final_image = np.zeros_like(test_truth)

        print 'TESTCASE_' + str(case_num).zfill(3) + '_MODALITY_' + str(modality_num) + '.nii.gz'

        for rep_idx in xrange(repetitions):

            print 'REPETITION..', rep_idx

            offset_slice = [slice(None)]*2 + [slice(rep_idx, None, 1)] * (test_data.ndim - 2)
            repatched_image = np.zeros_like(test_truth[offset_slice])
            corners_list = patchify_image(test_data[offset_slice], [test_data[offset_slice].shape[1]] + list(patch_shape))

            for corner_list_idx in xrange(0, len(corners_list), test_batch):

                corner_batch = corners_list[corner_list_idx:corner_list_idx+test_batch]
                input_patches = grab_patch(test_data[offset_slice], corners_list[corner_list_idx:corner_list_idx+test_batch], patch_shape)
                prediction = model.predict(input_patches)

                for corner_idx, corner in enumerate(corner_batch):
                    insert_patch(repatched_image, prediction[corner_idx, ...], corner)

            if rep_idx == 0:
                final_image = np.copy(repatched_image)
            else:
                final_image[offset_slice] = final_image[offset_slice] + (1.0 / (rep_idx)) * (repatched_image - final_image[offset_slice])
        
        print np.sum(final_image)

        final_image = np.around(np.squeeze(final_image))
        # final_image[final_image < 3] == 0

        save_numpy_2_nifti(final_image, output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_PREDICT.nii.gz'))

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