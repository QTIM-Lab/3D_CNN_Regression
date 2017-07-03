def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])

def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size, output_shape=compute_level_output_shape(filters=nb_filters, depth=depth, pool_size=pool_size, image_shape=image_shape), strides=strides, input_shape=compute_level_output_shape(filters=nb_filters, depth=depth+1, pool_size=pool_size, image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def fetch_data_files(data_dir, modality_dict, spreadsheet_dict=None, value_dict=None):

    """ This function gathers all modalities plus groundtruth for a given patient into
        a tuple, and then creates a list of tuples representing a list of all patients. Currently, groundtruth is just coded as "the last" file in a tuple for each case. In the future, maybe make two tuples for inputs and outputs.

        TODO: Smart error-catching.
    """

    print os.path.join(data_dir, "*")

    data_groups = {}

    # Get file modalities
    for subject_dir in glob.glob(os.path.join(data_dir, "*/")):

        for modality_group in modality_dict:

            data_groups[modality_group] = []
            modality_group_files = []

            for modality in modality_dict[modality_group]:
                target_file = glob.glob(os.path.join(subject_dir, modality))
                try:
                    modality_group_files.append(target_file[0])
                except:
                    break
            if len(modality_group_files) == len(modality_dict[modality_group]):
                data_groups[modality_group].append(tuple(modality_group_files))

    # TODO: Add section for spreadsheets.
    # TODO: Add section for values.
    #####

    return data_groups

def write_data_to_file(data_groups, output_hdf5_filepath, augmentation_dicts=None):

    """ Writes a list of filenames to an hdf5 file, subject to data augmentations. Data
        files can acquired using fetch_data_files.
    """

    example_augmentation_dict = [{'augmentation_functions': test_func,
                                    'params': [1,2],
                                    'augmentation_multiplier': 1,
                                    'augmentation_total': None,
                                    'augmentation_output_shapes': None}]

    # Get number of cases from an arbitrary data group.
    num_cases = len(data_groups[data_groups.keys()[0]])
    # Get modality numbers from all groups. This will currently fail for spreadsheet/value groups.
    modality_num = {}
    for key in data_groups:
        modality_nums[key] = len(data_groups[key][0])

    # Get output size for list of augmentations.
    for augmentation in augmentation_dict:

        # Get output rows
        try:
            multiplier = augmentation['augmentation_multiplier']
        except:
            multiplier = 1
        # Get data amount / cap
        try:
            total = augmentation['augmentation_total']
        except:
            total = None
        # If multiplier goes over "total", use total
        if total is None:
            num_cases *= multiplier
        elif ((num_cases * multiplier) - num_cases) > total:
            num_cases += total
        else:
            num_cases *= multiplier

        # Get output shape
        try:
            input_shape = augmentation['augmentation_input_shape']
        except:
            pass

    # Create Data File
    try:
        hdf5_file, data_storage, truth_storage = create_image_hdf5_file(output_hdf5_filepath, input_modality_num=input_modality_num, groundtruth_modality_num=groundtruth_modality_num, row_num=num_cases, input_shape=input_shape, groundtruth_shape=groundtruth_shape)
    except Exception as e:
        os.remove(output_hdf5_filepath)
        raise e

    # Write data
    write_image_data_to_hdf5(input_data_files, input_groundtruth_files, data_storage, truth_storage, image_shape, groundtruth_shape, augmentation_dicts)

    hdf5_file.close()
    return output_hdf5_filepath

def create_image_hdf5_file(output_filepath, input_modality_num, groundtruth_modality_num, row_num, input_shape, groundtruth_shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    # Input data has multiple 'channels' i.e. modalities.
    data_shape = tuple([0, input_modality_num] + list(input_shape))

    # Ground truth data only has one channel, ground truth modality.
    truth_shape = tuple([0, groundtruth_modality_num] + list(groundtruth_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=row_num)

    # For the classification case, dtype is UInt8Atom. For regression, we do Float again.
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape, filters=filters, expectedrows=row_num)

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
            data_storage.append(patch_data[:input_modality_num][np.newaxis])
            truth_storage.append(np.asarray(patch_data[input_modality_num:][np.newaxis]))

    return data_storage, truth_storage

def get_training_and_validation_generators(training_data_file, validation_data_file, batch_size, training_keys_file, validation_keys_file, train_test_split=0.8, overwrite=False):

    if validation_data_file:
        training_generator = data_generator(training_data_file, range(training_data_file.root.data.shape[0]), batch_size=batch_size)
        validation_generator = val_data_generator(validation_data_file, range(validation_data_file.root.data.shape[0]), batch_size=1)
        num_training_steps = training_data_file.root.data.shape[0] // batch_size
        num_validation_steps = validation_data_file.root.data.shape[0]

    else:
        training_list, validation_list = get_validation_split(training_data_file, data_split=train_test_split, overwrite=overwrite, training_file=training_keys_file, testing_file=validation_keys_file)
        training_generator = data_generator(training_data_file, training_list, batch_size=batch_size)
        validation_generator = data_generator(training_data_file, validation_list, batch_size=1)
        num_training_steps = len(training_list) // batch_size
        num_validation_steps = len(validation_list)    

    return training_generator, validation_generator, num_training_steps, num_validation_steps

def get_validation_split(data_file, training_file, testing_file, data_split=0.8, overwrite=False):

    if overwrite or not os.path.exists(training_file):

        num_cases = data_file.root.data.shape[0]
        sample_list = list(range(num_cases))

        training_list, testing_list = split_list(sample_list, split=data_split)

        pickle_dump(training_list, training_file)
        pickle_dump(testing_list, testing_file)

        return training_list, testing_list

    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(testing_file)


def split_list(input_list, split=0.8, shuffle_list=True):

    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

def test_func():
    return

def get_multi_class_labels(data, n_labels, labels=None):

    """ Not quite sure what's going on here, will investigate.
    """

    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y

def convert_data(x_list, y_list):

    """ This function once did more; will keep for now.
    """

    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y