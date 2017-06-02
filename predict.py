
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

def run_validation_case_patches(output_dir, model_object, model_file, data_file, patch_shape):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # model = load_old_model(model_file)
    model = model_object

    for case_num in xrange(data_file.root.data.shape[0]):

        test_data = np.asarray([data_file.root.data[case_num]])
        test_truth = np.asarray([data_file.root.truth[case_num]])

        save_numpy_2_nifti(np.squeeze(test_truth), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_TRUTH.nii.gz'))

        patched_test_data = patchify_image(test_data, patch_shape)
        repatched_image = np.zeros_like(test_truth)

        for corner, single_patch in patched_test_data:

            prediction = model.predict(single_patch)
            insert_patch(repatched_image, prediction, corner)

        save_numpy_2_nifti(np.squeeze(repatched_image), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_PREDICT.nii.gz'))

    data_file.close() 

def patchify_image(input_data, patch_shape):

    """ VERY wonky
    """

    print input_data.shape
    # fd 

    patch_shape = [input_data.shape[1]] + list(patch_shape)
    patch_list = []
    corner = [0] * len(input_data.shape[1:])
    
    finished = False

    while not finished:

        patch_slice = [slice(None)] + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]
        patch = input_data[patch_slice]
        patch_list += [[corner[:], patch[:]]]

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

            # print corner

    # print patch_list
    return patch_list

def insert_patch(input_data, patch, corner):

    patch_shape = patch.shape[1:]
    patch_slice = [slice(None)] + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]
    input_data[patch_slice] = patch

    return

def calculate_prediction_msq(output_dir):

    """ Calculate mean-squared error for the predictions folder.
    """

if __name__ == '__main__':
    pass