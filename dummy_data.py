import os
import numpy as np

from image_utils import save_numpy_2_nifti

def dummy_data_generator(cases=50, input_modalities=4, modality_dims=(5,5,5), dummy_data_folder = './dummy_data', train_test = [50, 10]):

    dummy_data_folder = os.path.abspath(dummy_data_folder)

    if not os.path.exists(dummy_data_folder):
        os.mkdir(dummy_data_folder)

    folder_labels = [[train_test[0], 'train'], [train_test[1], 'test']]

    for cases, folder_label in folder_labels:

        dummy_data_subfolder = os.path.join(dummy_data_folder, folder_label)
        if not os.path.exists(dummy_data_subfolder):
            os.mkdir(dummy_data_subfolder)

        for case_num in xrange(cases):

            output_folder = os.path.join(dummy_data_subfolder, 'CASE_' + str(case_num).zfill(3))

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            case = []
            output_groundtruth = np.zeros(modality_dims)

            for mod_num in xrange(input_modalities):

                modality = np.random.random(modality_dims)
                save_numpy_2_nifti(modality, '', os.path.join(output_folder, 'modality_' + str(mod_num) + '.nii.gz'))
                output_groundtruth = output_groundtruth + modality

            output_groundtruth = output_groundtruth / input_modalities

            save_numpy_2_nifti(output_groundtruth, '', os.path.join(output_folder, 'groundtruth.nii.gz'))

if __name__ == '__main__':
    dummy_data_generator()