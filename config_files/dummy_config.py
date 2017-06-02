import os

config = dict()

# Data will be compressed in hdf5 format at these filepaths.
config["train_dir"] = os.path.abspath("./dummy_data/train")
config["test_dir"] = os.path.abspath("./dummy_data/test")
config["hdf5_train"] = './hdf5_data/dummy_train.hdf5'
config["hdf5_test"] = './hdf5_data/dummy_test.hdf5'

# Image Information
config["image_shape"] = (100, 100, 100)

# Patch Information
config['patches'] = True
config['patch_shape'] = (8, 8, 8)
config['train_patch_num'] = 2000

# Modalities. Always make input_groundtruth as list.
config["input_modalities"] = ["modality_0", "modality_1", "modality_2", "modality_3"]
config["input_groundtruth"] = ['groundtruth']

# Path to save model.
config["model_file"] = os.path.abspath("./model_files/dummy_model.h5")

# Model parameters
config["downsize_filters_factor"] = 1
config["decay_learning_rate_every_x_epochs"] = 10
config["initial_learning_rate"] = 0.001
config["learning_rate_drop"] = 0.9
config["n_epochs"] = 50

# Model training parameters
config["train_test_split"] = .8
config["batch_size"] = 10
config["training_file"] = os.path.abspath("./hdf5_data/training_ids.pkl")
config["validation_file"] = os.path.abspath("./hdf5_data/validation_ids.pkl")

# Model testing parameters
config['predictions_folder'] = os.path.abspath('./predictions')

# config["pool_size"] = (2, 2, 2)  # This determines what shape the images will be cropped/resampled to.
# config["n_labels"] = 1  # not including background
# config["n_epochs"] = 50
# config["decay_learning_rate_every_x_epochs"] = 10
# config["initial_learning_rate"] = 0.00001
# config["learning_rate_drop"] = 0.5
# config["validation_split"] = 0.8
# config["smooth"] = 1.0
# # config["nb_channels"] = len(config["training_modalities"])
# # config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
# config["truth_channel"] = config["nb_channels"]
# config["background_channel"] = config["nb_channels"] + 1
# config["deconvolution"] = False  # use deconvolution instead of up-sampling. Requires keras-contrib.
# # divide the number of filters used by by a given factor. This will reduce memory consumption.
