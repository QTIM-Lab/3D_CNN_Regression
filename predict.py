
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
    """
    Runs a test case and writes predicted images to file.
    :param test_index: Index from of the list of test cases to get an image prediction from.  
    :param out_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_old_model(model_file)

    for case_num in xrange(data_file.root.data.shape[0]):
        test_data = np.asarray([data_file.root.data[case_num]])
        test_truth = np.asarray([data_file.root.truth[case_num]])

        save_numpy_2_nifti(np.squeeze(test_truth), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_TRUTH.nii.gz'))
        prediction = model.predict(test_data)
        print prediction.shape
        save_numpy_2_nifti(np.squeeze(prediction), output_filepath=os.path.join(output_dir, 'TESTCASE_' + str(case_num).zfill(3) + '_PREDICT.nii.gz'))

        # for i, modality in enumerate(training_modalities):
        #     image = nib.Nifti1Image(test_data[0, i], affine)
        #     image.to_filename(os.path.join(out_dir, "data_{0}.nii.gz".format(modality)))

        # test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
        # test_truth.to_filename(os.path.join(out_dir, "truth.nii.gz"))

    # fd = dg

    # prediction = model.predict(test_data)
    # prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold, labels=labels)
    # if isinstance(prediction_image, list):
    #     for i, image in enumerate(prediction_image):
    #         image.to_filename(os.path.join(out_dir, "prediction_{0}.nii.gz".format(i + 1)))
    # else:
    #     prediction_image.to_filename(os.path.join(out_dir, "prediction.nii.gz"))

    data_file.close()

if __name__ == '__main__':
    pass