""" Most of these functions taken from the qtim_tools python package.

TODO: Accept many formats.
"""

import nibabel as nib
import numpy as np

def generate_identity_affine(timepoints=1):

    """ A convenient function for generating an identity affine matrix. Can be
        used for saving blank niftis.
    """

    # Needlessly complicated.

    if timepoints == 1:
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    else:
        return np.swapaxes(np.tile(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (timepoints, 1,1)), 0, 2)

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath='', output_filepath=[]):

    if reference_nifti_filepath != '':
        nifti_image = nib.load(reference_nifti_filepath)
        image_affine = nifti_image.affine
    else:
        # print 'Warning: no reference nifti file provided. Generating empty header.'
        image_affine = generate_identity_affine()

    output_nifti = nib.Nifti1Image(image_numpy, image_affine)

    if output_filepath == []:
        return output_nifti
    else:
        nib.save(output_nifti, output_filepath)

def nifti_2_numpy(filepath):

    """ There are a lot of repetitive conversions in the current iteration
        of this program. Another option would be to always pass the nibabel
        numpy class, which contains image and attributes. But not everyone
        knows how to use that class, so it may be more difficult to troubleshoot.
    """

    img = nib.load(filepath).get_data().astype(float)
    return img
