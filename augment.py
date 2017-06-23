
class Augmentation(object):

    def __init__(self, multiplier=None, total=None, output_shape=None):

        # Note: total feature is as of yet unimplemented..

        self.multiplier = multiplier
        self.total = total
        self.output_shape = output_shape

        return

class ExtractPatches(Augmentation):

    def __init__(self, patch_shape, patch_extraction_conditions):

        self.patch_shape = patch_shape
        self.patch_extraction_conditions = self.patch_extraction_conditions

    def augment(self, input_data):

        output_data = [input_data]

        if multiplier == 1:
            return output_data

        # Patch extraction conditions
        start_idx = 0
        condition_list = [-1] * (multiplier - 1)
        for condition_idx, patch_extraction_condition in enumerate(patch_extraction_conditions):
            end_idx = start_idx + int(np.ceil(patch_extraction_condition[1]*patches_per_image))
            condition_list[start_idx:end_idx] = [condition_idx]*(end_idx-start_idx)
            start_idx = end_idx

        print 'PATCH TYPES: ', condition_list

        for condition in condition_list:

            # Extract patch..
            if condition < 0:
                output_data += [extract_patch(input_data)]
            else:
                output_data = [extract_patch(input_data, condition)]

        return output_data

    def extract_patch(self, input_data, patch_extraction_condition_idx=None):

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

            corner = [np.random.randint(0, max_dim) for max_dim in input_data.shape[1:]]
            patch_slice = [slice(None)] + [slice(corner_dim, corner_dim+self.patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]
            patch = input_image_stack[patch_slice]

            pad_dims = [(0,0)]
            for idx, dim in enumerate(patch.shape[1:]):
                pad_dims += [(0, self.patch_shape[idx]-dim)]

            patch = np.lib.pad(patch, tuple(pad_dims), 'edge')

            if patch_extraction_condition_idx is not None:
                acceptable_patch = self.patch_extraction_conditions[patch_extraction_condition_idx](patch)
            else:
                acceptable_patch = True

        return patch

if __name__ == '__main__':
    pass